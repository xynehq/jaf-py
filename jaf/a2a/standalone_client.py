"""
Standalone A2A client that can be used independently of the JAF framework.
This module provides a complete A2A client implementation without relative imports.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field


# Standalone types for the client
class StandaloneA2AClientConfig(BaseModel):
    """A2A client configuration"""

    model_config = {"frozen": True}

    base_url: str = Field(alias="baseUrl")
    timeout: Optional[int] = None


class StandaloneA2AClientState(BaseModel):
    """A2A client state"""

    model_config = {"frozen": True}

    config: StandaloneA2AClientConfig
    session_id: str = Field(alias="sessionId")


# Client functions
def create_standalone_a2a_client(
    base_url: str, config: Optional[Dict[str, Any]] = None
) -> StandaloneA2AClientState:
    """Pure function to create standalone A2A client"""
    config = config or {}

    return StandaloneA2AClientState(
        config=StandaloneA2AClientConfig(
            baseUrl=base_url.rstrip("/"),  # Remove trailing slash
            timeout=config.get("timeout", 30000),
        ),
        sessionId=f"client_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    )


def create_standalone_message_request(
    message: str, session_id: str, configuration: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pure function to create message request"""
    return {
        "jsonrpc": "2.0",
        "id": f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
                "contextId": session_id,
                "kind": "message",
            },
            "configuration": configuration,
        },
    }


def create_standalone_streaming_message_request(
    message: str, session_id: str, configuration: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Pure function to create streaming message request"""
    return {
        "jsonrpc": "2.0",
        "id": f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
                "contextId": session_id,
                "kind": "message",
            },
            "configuration": configuration,
        },
    }


async def send_standalone_http_request(
    url: str, body: Dict[str, Any], timeout: int = 30000
) -> Dict[str, Any]:
    """Pure function to send HTTP request"""
    timeout_seconds = timeout / 1000.0

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.post(
                url,
                json=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )

            if not response.is_success:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            return response.json()

        except httpx.TimeoutException:
            raise Exception(f"Request timeout after {timeout}ms")


async def send_standalone_a2a_request(
    client: StandaloneA2AClientState, request: Dict[str, Any]
) -> Dict[str, Any]:
    """Pure function to send A2A request"""
    url = f"{client.config.base_url}/a2a"
    return await send_standalone_http_request(url, request, client.config.timeout)


async def send_standalone_message(
    client: StandaloneA2AClientState, message: str, configuration: Optional[Dict[str, Any]] = None
) -> str:
    """Pure function to send message"""
    request = create_standalone_message_request(message, client.session_id, configuration)
    response = await send_standalone_a2a_request(client, request)

    if "error" in response:
        error = response["error"]
        raise Exception(f"A2A Error {error['code']}: {error['message']}")

    return extract_standalone_text_response(response.get("result"))


async def stream_standalone_message(
    client: StandaloneA2AClientState, message: str, configuration: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Pure function to stream message"""
    request = create_standalone_streaming_message_request(message, client.session_id, configuration)
    url = f"{client.config.base_url}/a2a"

    timeout_seconds = client.config.timeout / 1000.0

    async with httpx.AsyncClient(timeout=timeout_seconds) as http_client:
        try:
            async with http_client.stream(
                "POST",
                url,
                json=request,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            ) as response:
                if not response.is_success:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop() or ""

                    for line in lines:
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data.strip():
                                try:
                                    event = json.loads(data)
                                    if "result" in event:
                                        yield event["result"]
                                except json.JSONDecodeError:
                                    print(f"Failed to parse SSE data: {data}")
                                    continue

        except httpx.TimeoutException:
            raise Exception(f"Stream timeout after {client.config.timeout}ms")


async def get_standalone_agent_card(client: StandaloneA2AClientState) -> Dict[str, Any]:
    """Pure function to get agent card"""
    url = f"{client.config.base_url}/.well-known/agent-card"

    async with httpx.AsyncClient(timeout=client.config.timeout / 1000.0) as http_client:
        response = await http_client.get(url, headers={"Accept": "application/json"})

        if not response.is_success:
            raise Exception(f"Failed to get agent card: HTTP {response.status_code}")

        return response.json()


async def send_standalone_message_to_agent(
    client: StandaloneA2AClientState,
    agent_name: str,
    message: str,
    configuration: Optional[Dict[str, Any]] = None,
) -> str:
    """Pure function to send message to specific agent"""
    request = create_standalone_message_request(message, client.session_id, configuration)
    url = f"{client.config.base_url}/a2a/agents/{agent_name}"

    response = await send_standalone_http_request(url, request, client.config.timeout)

    if "error" in response:
        error = response["error"]
        raise Exception(f"A2A Error {error['code']}: {error['message']}")

    return extract_standalone_text_response(response.get("result"))


def extract_standalone_text_response(result: Any) -> str:
    """Pure function to extract text response"""
    # Handle direct string response
    if isinstance(result, str):
        return result

    # Handle task response
    if isinstance(result, dict) and result.get("kind") == "task":
        # Extract from artifacts
        artifacts = result.get("artifacts", [])
        if artifacts:
            text_artifact = None
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                text_parts = [part for part in parts if part.get("kind") == "text"]
                if text_parts:
                    text_artifact = artifact
                    break

            if text_artifact:
                text_parts = [part for part in text_artifact["parts"] if part.get("kind") == "text"]
                if text_parts:
                    return text_parts[0].get("text", "No text content")

        # Extract from history
        history = result.get("history", [])
        if history:
            last_message = history[-1]
            parts = last_message.get("parts", [])
            text_parts = [part.get("text") for part in parts if part.get("kind") == "text"]
            if text_parts:
                return "\n".join(filter(None, text_parts))

        return "Task completed but no text response available"

    # Handle message response
    if isinstance(result, dict) and result.get("kind") == "message":
        parts = result.get("parts", [])
        text_parts = [part.get("text") for part in parts if part.get("kind") == "text"]
        if text_parts:
            return "\n".join(filter(None, text_parts))

    # Handle object responses
    if isinstance(result, dict):
        return json.dumps(result, indent=2)

    return "No response content available"


async def connect_to_standalone_a2a_agent(base_url: str) -> Dict[str, Any]:
    """Pure function to connect to A2A agent (convenience function)"""
    client = create_standalone_a2a_client(base_url)
    agent_card = await get_standalone_agent_card(client)

    async def ask(message: str, config: Optional[Dict[str, Any]] = None) -> str:
        return await send_standalone_message(client, message, config)

    async def stream(message: str, config: Optional[Dict[str, Any]] = None):
        async for event in stream_standalone_message(client, message, config):
            yield event

    async def ask_agent(
        agent_name: str, message: str, config: Optional[Dict[str, Any]] = None
    ) -> str:
        return await send_standalone_message_to_agent(client, agent_name, message, config)

    return {
        "client": client,
        "agent_card": agent_card,
        "ask": ask,
        "stream": stream,
        "ask_agent": ask_agent,
    }


# Simple usage example
async def main_example():
    """Example usage of standalone A2A client"""
    try:
        # Connect to A2A server
        connection = await connect_to_standalone_a2a_agent("http://localhost:3000")

        print("Connected to A2A server!")
        print(
            f"Available agents: {[skill['name'] for skill in connection['agent_card']['skills']]}"
        )

        # Send a message
        response = await connection["ask"]("Hello, how can you help me?")
        print(f"Response: {response}")

        # Send message to specific agent
        if "ask_agent" in connection:
            math_response = await connection["ask_agent"]("MathTutor", "What is 5 + 3?")
            print(f"Math response: {math_response}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main_example())
