"""
Pure functional A2A client
All client operations are pure functions using httpx
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Dict, Optional

import httpx

from .types import (
    A2AClientConfig,
    A2AClientState,
)


def create_a2a_client(base_url: str, config: Optional[Dict[str, Any]] = None) -> A2AClientState:
    """Pure function to create A2A client"""
    config = config or {}

    return A2AClientState(
        config=A2AClientConfig(
            baseUrl=base_url.rstrip("/"),  # Remove trailing slash
            timeout=config.get("timeout", 30000)
        ),
        sessionId=f"client_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    )


def create_message_request(
    message: str,
    session_id: str,
    configuration: Optional[Dict[str, Any]] = None
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
                "kind": "message"
            },
            "configuration": configuration
        }
    }


def create_streaming_message_request(
    message: str,
    session_id: str,
    configuration: Optional[Dict[str, Any]] = None
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
                "kind": "message"
            },
            "configuration": configuration
        }
    }


async def send_http_request(
    url: str,
    body: Dict[str, Any],
    timeout: int = 30000
) -> Dict[str, Any]:
    """Pure function to send HTTP request"""
    timeout_seconds = timeout / 1000.0

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.post(
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )

            if not response.is_success:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            return response.json()

        except httpx.TimeoutException:
            raise Exception(f"Request timeout after {timeout}ms")


async def send_a2a_request(
    client: A2AClientState,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Pure function to send A2A request"""
    url = f"{client.config.base_url}/a2a"
    return await send_http_request(url, request, client.config.timeout)


async def send_message(
    client: A2AClientState,
    message: str,
    configuration: Optional[Dict[str, Any]] = None
) -> str:
    """Pure function to send message"""
    request = create_message_request(message, client.session_id, configuration)
    response = await send_a2a_request(client, request)

    if "error" in response:
        error = response["error"]
        raise Exception(f"A2A Error {error['code']}: {error['message']}")

    return extract_text_response(response.get("result"))


async def stream_message(
    client: A2AClientState,
    message: str,
    configuration: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Pure function to stream message"""
    request = create_streaming_message_request(message, client.session_id, configuration)
    url = f"{client.config.base_url}/a2a"

    timeout_seconds = client.config.timeout / 1000.0

    async with httpx.AsyncClient(timeout=timeout_seconds) as http_client:
        try:
            async with http_client.stream(
                "POST",
                url,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                }
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


async def get_agent_card(client: A2AClientState) -> Dict[str, Any]:
    """Pure function to get agent card"""
    url = f"{client.config.base_url}/.well-known/agent-card"

    async with httpx.AsyncClient(timeout=client.config.timeout / 1000.0) as http_client:
        response = await http_client.get(
            url,
            headers={"Accept": "application/json"}
        )

        if not response.is_success:
            raise Exception(f"Failed to get agent card: HTTP {response.status_code}")

        return response.json()


async def discover_agents(base_url: str) -> Dict[str, Any]:
    """Pure function to discover agents"""
    client = create_a2a_client(base_url)
    return await get_agent_card(client)


async def send_message_to_agent(
    client: A2AClientState,
    agent_name: str,
    message: str,
    configuration: Optional[Dict[str, Any]] = None
) -> str:
    """Pure function to send message to specific agent"""
    request = create_message_request(message, client.session_id, configuration)
    url = f"{client.config.base_url}/a2a/agents/{agent_name}"

    response = await send_http_request(url, request, client.config.timeout)

    if "error" in response:
        error = response["error"]
        raise Exception(f"A2A Error {error['code']}: {error['message']}")

    return extract_text_response(response.get("result"))


async def stream_message_to_agent(
    client: A2AClientState,
    agent_name: str,
    message: str,
    configuration: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Pure function to stream message to specific agent"""
    request = create_streaming_message_request(message, client.session_id, configuration)
    url = f"{client.config.base_url}/a2a/agents/{agent_name}"

    timeout_seconds = client.config.timeout / 1000.0

    async with httpx.AsyncClient(timeout=timeout_seconds) as http_client:
        try:
            async with http_client.stream(
                "POST",
                url,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                }
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
                            data = line[6:]
                            if data.strip():
                                try:
                                    event = json.loads(data)
                                    if "result" in event:
                                        yield event["result"]
                                except json.JSONDecodeError:
                                    continue

        except httpx.TimeoutException:
            raise Exception(f"Stream timeout after {client.config.timeout}ms")


def extract_text_response(result: Any) -> str:
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


async def check_a2a_health(client: A2AClientState) -> Dict[str, Any]:
    """Pure function to check A2A server health"""
    url = f"{client.config.base_url}/a2a/health"

    async with httpx.AsyncClient(timeout=client.config.timeout / 1000.0) as http_client:
        response = await http_client.get(
            url,
            headers={"Accept": "application/json"}
        )

        if not response.is_success:
            raise Exception(f"Health check failed: HTTP {response.status_code}")

        return response.json()


async def get_a2a_capabilities(client: A2AClientState) -> Dict[str, Any]:
    """Pure function to get A2A capabilities"""
    url = f"{client.config.base_url}/a2a/capabilities"

    async with httpx.AsyncClient(timeout=client.config.timeout / 1000.0) as http_client:
        response = await http_client.get(
            url,
            headers={"Accept": "application/json"}
        )

        if not response.is_success:
            raise Exception(f"Capabilities request failed: HTTP {response.status_code}")

        return response.json()


async def connect_to_a2a_agent(base_url: str) -> Dict[str, Any]:
    """Pure function to connect to A2A agent (convenience function)"""
    client = create_a2a_client(base_url)
    agent_card = await get_agent_card(client)

    async def ask(message: str, config: Optional[Dict[str, Any]] = None) -> str:
        return await send_message(client, message, config)

    async def stream(message: str, config: Optional[Dict[str, Any]] = None):
        async for event in stream_message(client, message, config):
            yield event

    async def health():
        return await check_a2a_health(client)

    async def capabilities():
        return await get_a2a_capabilities(client)

    return {
        "client": client,
        "agent_card": agent_card,
        "ask": ask,
        "stream": stream,
        "health": health,
        "capabilities": capabilities
    }


# Utility functions

def create_a2a_message_dict(
    text: str,
    role: str = "user",
    context_id: Optional[str] = None
) -> Dict[str, Any]:
    """Utility function to create A2A message dictionary"""
    return {
        "role": role,
        "parts": [{"kind": "text", "text": text}],
        "messageId": f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "contextId": context_id,
        "kind": "message"
    }


def parse_sse_event(line: str) -> Optional[Dict[str, Any]]:
    """Pure function to parse Server-Sent Event line"""
    if not line.startswith("data: "):
        return None

    data = line[6:]  # Remove "data: " prefix
    if not data.strip():
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def validate_a2a_response(response: Dict[str, Any]) -> bool:
    """Pure function to validate A2A response"""
    return (
        isinstance(response, dict) and
        response.get("jsonrpc") == "2.0" and
        "id" in response and
        ("result" in response or "error" in response)
    )
