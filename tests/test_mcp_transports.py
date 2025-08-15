import asyncio
import json
from typing import Dict, Any

import pytest
import respx
from httpx import Response

from jaf.providers.mcp import (
    StreamableHttpMCPTransport,
    SSEMCPTransport,
)

@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_mcp_transport_send_request():
    """Test sending a request with StreamableHttpMCPTransport."""
    uri = "http://test-server/mcp"
    transport = StreamableHttpMCPTransport(uri)
    await transport.connect()

    # Mock the HTTP response
    mock_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"status": "success"}
    }
    respx.post(uri).mock(return_value=Response(200, json=mock_response))

    # Send a request
    response = await transport.send_request("test_method", {"param": "value"})

    # Assertions
    assert response == mock_response
    await transport.disconnect()

@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_mcp_transport_send_request_error():
    """Test handling of HTTP errors when sending a request."""
    uri = "http://test-server/mcp"
    transport = StreamableHttpMCPTransport(uri)
    await transport.connect()

    # Mock an HTTP error response
    respx.post(uri).mock(return_value=Response(500))

    # Send a request and expect a RuntimeError
    with pytest.raises(RuntimeError, match="HTTP request failed"):
        await transport.send_request("test_method", {"param": "value"})

    await transport.disconnect()

@pytest.mark.asyncio
@respx.mock
async def test_streamable_http_mcp_transport_send_notification():
    """Test sending a notification with StreamableHttpMCPTransport."""
    uri = "http://test-server/mcp"
    transport = StreamableHttpMCPTransport(uri)
    await transport.connect()

    # Mock the HTTP response
    respx.post(uri).mock(return_value=Response(204))

    # Send a notification
    await transport.send_notification("test_notification", {"param": "value"})

    await transport.disconnect()

@pytest.mark.asyncio
async def test_sse_mcp_transport_unsupported_methods():
    """Test that SSE transport raises errors for unsupported methods."""
    uri = "http://test-server/sse"
    transport = SSEMCPTransport(uri)

    with pytest.raises(NotImplementedError):
        await transport.send_request("test_method", {})

    with pytest.raises(NotImplementedError):
        await transport.send_notification("test_notification", {})

@pytest.mark.asyncio
@respx.mock
async def test_sse_mcp_transport_listen():
    """Test the SSE transport's listener."""
    uri = "http://test-server/sse"
    transport = SSEMCPTransport(uri)

    # Mock the SSE stream
    sse_data = [
        'event: message\n',
        'data: {"key": "value1"}\n',
        '\n',
        'event: notification\n',
        'data: {"key": "value2"}\n',
        '\n',
    ]
    respx.get(uri).mock(return_value=Response(200, text="".join(sse_data), headers={"Content-Type": "text/event-stream"}))

    # Connect and listen
    await transport.connect()
    await asyncio.sleep(0.1)  # Allow the listener to process events
    await transport.disconnect()
