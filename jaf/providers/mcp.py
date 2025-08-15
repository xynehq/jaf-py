"""
Model Context Protocol (MCP) provider implementation for JAF.

This module provides MCP client functionality to integrate with external
tools and services following the MCP specification.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar

import httpx
import websockets
from httpx_sse import aconnect_sse
from pydantic import BaseModel

from ..core.tool_results import ToolErrorCodes, ToolResult, ToolResultStatus
from ..core.types import ToolSchema

Ctx = TypeVar('Ctx')

@dataclass(frozen=True)
class MCPCapabilities:
    """MCP server capabilities."""
    tools: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class MCPClientInfo:
    """MCP client information."""
    name: str
    version: str

@dataclass(frozen=True)
class MCPServerInfo:
    """MCP server information."""
    name: str
    version: str
    capabilities: MCPCapabilities = field(default_factory=MCPCapabilities)

class MCPTransport(ABC):
    """Abstract base class for MCP transport mechanisms."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass

    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and return the response."""
        pass

    @abstractmethod
    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification (no response expected)."""
        pass

class WebSocketMCPTransport(MCPTransport):
    """WebSocket-based MCP transport."""

    def __init__(self, uri: str):
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}

    async def connect(self) -> None:
        """Connect to the WebSocket server."""
        self.websocket = await websockets.connect(self.uri)
        # Start listening for responses
        asyncio.create_task(self._listen())

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self.websocket:
            raise RuntimeError("Not connected to MCP server")

        request_id = self._request_id
        self._request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Create future for response
        future = asyncio.Future()
        self._pending_requests[request_id] = future

        # Send request
        await self.websocket.send(json.dumps(request))

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        finally:
            self._pending_requests.pop(request_id, None)

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification."""
        if not self.websocket:
            raise RuntimeError("Not connected to MCP server")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        await self.websocket.send(json.dumps(notification))

    async def _listen(self) -> None:
        """Listen for incoming messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    if "id" in data and data["id"] in self._pending_requests:
                        # It's a response to a request
                        future = self._pending_requests[data["id"]]
                        if "error" in data:
                            future.set_exception(Exception(f"MCP Error: {data['error']}"))
                        else:
                            future.set_result(data.get("result", {}))
                except json.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            pass

class SSEMCPTransport(MCPTransport):
    """SSE-based MCP transport."""

    def __init__(self, uri: str):
        self.uri = uri
        self.client: Optional[httpx.AsyncClient] = None
        self.sse_connection = None

    async def connect(self) -> None:
        """Connect to the SSE endpoint."""
        self.client = httpx.AsyncClient()
        print(f"Connecting to SSE endpoint at {self.uri}...")
        self.sse_connection = await aconnect_sse(self.client, "GET", self.uri)
        asyncio.create_task(self._listen())
        print("SSE connection established.")

    async def disconnect(self) -> None:
        """Disconnect from the SSE endpoint."""
        if self.client:
            await self.client.aclose()
            self.client = None
        print("SSE connection closed.")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request over SSE (not supported for this transport)."""
        raise NotImplementedError("SSE transport does not support client-to-server requests.")

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification over SSE (not supported for this transport)."""
        raise NotImplementedError("SSE transport does not support client-to-server notifications.")

    async def _listen(self) -> None:
        """Listen for and print incoming SSE events."""
        print("SSE transport listening for events...")
        try:
            async for event in self.sse_connection.aiter_sse():
                print(f"[SSE Event] type={event.event}, data={event.data}")
        except httpx.ConnectError as e:
            print(f"SSE connection error: {e}")
        except Exception as e:
            print(f"An error occurred in the SSE listener: {e}")


class StreamableHttpMCPTransport(MCPTransport):
    """Streamable HTTP-based MCP transport."""

    def __init__(self, uri: str):
        self.uri = uri
        self.client: Optional[httpx.AsyncClient] = None
        self._request_id = 0

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        self.client = httpx.AsyncClient()

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send an HTTP request and return the JSON response."""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }

        try:
            response = await self.client.post(self.uri, json=request, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON response: {e}")

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send an HTTP notification (fire-and-forget)."""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        await self.client.post(self.uri, json=notification, timeout=10.0)


class StdioMCPTransport(MCPTransport):
    """Stdio-based MCP transport for local processes."""

    def __init__(self, command: List[str]):
        self.command = command
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Start the MCP server process."""
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Start reading responses
        self._reader_task = asyncio.create_task(self._read_responses())

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request."""
        if not self.process:
            raise RuntimeError("MCP server process not started")

        request_id = self._request_id
        self._request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Create future for response
        future = asyncio.Future()
        self._pending_requests[request_id] = future

        # Send request
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

        # Wait for response
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        finally:
            self._pending_requests.pop(request_id, None)

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification."""
        if not self.process:
            raise RuntimeError("MCP server process not started")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

    async def _read_responses(self) -> None:
        """Read responses from the server process."""
        try:
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                try:
                    data = json.loads(line.decode().strip())
                    if "id" in data and data["id"] in self._pending_requests:
                        future = self._pending_requests[data["id"]]
                        if "error" in data:
                            future.set_exception(Exception(f"MCP Error: {data['error']}"))
                        else:
                            future.set_result(data.get("result", {}))
                except json.JSONDecodeError:
                    continue
        except asyncio.CancelledError:
            pass

class MCPClient:
    """MCP client for interacting with MCP servers."""

    def __init__(self, transport: MCPTransport, client_info: MCPClientInfo):
        self.transport = transport
        self.client_info = client_info
        self.server_info: Optional[MCPServerInfo] = None
        self._tools: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize the MCP connection."""
        await self.transport.connect()

        # Send initialize request
        response = await self.transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": self.client_info.name,
                "version": self.client_info.version
            }
        })

        # Parse server info
        server_info = response.get("serverInfo", {})
        capabilities_data = response.get("capabilities", {})

        self.server_info = MCPServerInfo(
            name=server_info.get("name", "unknown"),
            version=server_info.get("version", "unknown"),
            capabilities=MCPCapabilities(**capabilities_data)
        )

        # Send initialized notification
        await self.transport.send_notification("notifications/initialized", {})

        # Load available tools
        await self._load_tools()

    async def close(self) -> None:
        """Close the MCP connection."""
        await self.transport.disconnect()

    async def _load_tools(self) -> None:
        """Load available tools from the server."""
        try:
            response = await self.transport.send_request("tools/list", {})
            tools = response.get("tools", [])

            for tool_info in tools:
                self._tools[tool_info["name"]] = tool_info
        except Exception as e:
            print(f"Failed to load MCP tools: {e}")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")

        response = await self.transport.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

        return response

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        return self._tools.get(name)

class MCPToolArgs(BaseModel):
    """Base class for MCP tool arguments."""
    pass

class MCPTool:
    """A tool that proxies to an MCP server."""

    def __init__(self, mcp_client: MCPClient, tool_name: str, args_model: type[BaseModel]):
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.args_model = args_model

        # Get tool info from MCP server
        tool_info = mcp_client.get_tool_info(tool_name)
        if not tool_info:
            raise ValueError(f"Tool '{tool_name}' not found on MCP server")

        self._schema = ToolSchema(
            name=tool_name,
            description=tool_info.get("description", f"MCP tool: {tool_name}"),
            parameters=args_model
        )

    @property
    def schema(self) -> ToolSchema[MCPToolArgs]:
        """Get the tool schema."""
        return self._schema

    async def execute(self, args: MCPToolArgs, context: Ctx) -> ToolResult:
        """Execute the MCP tool."""
        try:
            # Convert Pydantic model to dict
            if hasattr(args, 'model_dump'):
                args_dict = args.model_dump()
            elif hasattr(args, 'dict'):
                args_dict = args.dict()
            else:
                args_dict = dict(args)

            # Call the MCP tool
            response = await self.mcp_client.call_tool(self.tool_name, args_dict)

            # Check for errors in response
            if "error" in response:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error_code=ToolErrorCodes.EXECUTION_FAILED,
                    error_message=response["error"].get("message", "MCP tool execution failed"),
                    data=response
                )

            # Extract content from response
            content = response.get("content", [])
            if not content:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data="",
                    metadata={"mcp_response": response}
                )

            # Combine text content
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            result_text = "\n".join(text_parts) if text_parts else str(response)

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data=result_text,
                metadata={"mcp_response": response}
            )

        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_code=ToolErrorCodes.EXECUTION_FAILED,
                error_message=f"MCP tool execution failed: {e!s}",
                data={"error": str(e)}
            )

def create_mcp_websocket_client(uri: str, client_name: str = "JAF", client_version: str = "2.0.0") -> MCPClient:
    """Create an MCP client using WebSocket transport."""
    transport = WebSocketMCPTransport(uri)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info)

def create_mcp_stdio_client(command: List[str], client_name: str = "JAF", client_version: str = "2.0.0") -> MCPClient:
    """Create an MCP client using stdio transport."""
    transport = StdioMCPTransport(command)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info)


def create_mcp_sse_client(uri: str, client_name: str = "JAF", client_version: str = "2.0.0") -> MCPClient:
    """Create an MCP client using SSE transport."""
    transport = SSEMCPTransport(uri)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info)


def create_mcp_http_client(uri: str, client_name: str = "JAF", client_version: str = "2.0.0") -> MCPClient:
    """Create an MCP client using streamable HTTP transport."""
    transport = StreamableHttpMCPTransport(uri)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info)

async def create_mcp_tools_from_client(mcp_client: MCPClient) -> List[MCPTool]:
    """Create JAF tools from all available MCP tools."""
    await mcp_client.initialize()

    tools = []
    for tool_name in mcp_client.get_available_tools():
        # Create a generic args model for this tool
        class GenericMCPArgs(MCPToolArgs):
            pass

        tool = MCPTool(mcp_client, tool_name, GenericMCPArgs)
        tools.append(tool)

    return tools
