"""
Model Context Protocol (MCP) provider implementation for JAF.

This module provides MCP client functionality to integrate with external
tools and services following the MCP specification.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Union

import httpx
from httpx_sse import aconnect_sse
from pydantic import BaseModel, create_model

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
    experimental: Optional[Dict[str, Any]] = None

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


class SSEMCPTransport(MCPTransport):
    """SSE-based MCP transport."""

    def __init__(self, uri: str, timeout: float = 30.0):
        self.uri = uri
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._session_id: Optional[str] = None

    async def connect(self) -> None:
        """Connect to the SSE endpoint and start listening."""
        self.client = httpx.AsyncClient()
        self._listen_task = asyncio.create_task(self._listen())

    async def disconnect(self) -> None:
        """Disconnect from the SSE endpoint."""
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        if self.client:
            await self.client.aclose()
            self.client = None

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request over SSE using HTTP POST."""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params
        }

        # Create future for response
        future = asyncio.Future()
        self._pending_requests[self._request_id] = future

        try:
            # Wait a bit for session ID to be extracted if not available yet
            if not self._session_id:
                await asyncio.sleep(1.0)
            
            # Determine the correct endpoint for sending requests
            if self._session_id:
                # Extract base server URL (remove path) and add messages endpoint
                from urllib.parse import urlparse
                parsed = urlparse(self.uri)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                messages_url = f"{base_url}/messages/?session_id={self._session_id}"
            else:
                # Fallback to the base endpoint
                messages_url = self.uri
            
            # Send request via HTTP POST
            response = await self.client.post(messages_url, json=request, timeout=self.timeout)
            response.raise_for_status()
            
            # For SSE, we expect the response to come through the SSE stream
            # Wait for the response
            result = await asyncio.wait_for(future, timeout=self.timeout)
            return result
        except httpx.HTTPStatusError as e:
            self._pending_requests.pop(self._request_id, None)
            raise RuntimeError(f"HTTP request failed: {e}") from e
        except httpx.RequestError as e:
            self._pending_requests.pop(self._request_id, None)
            raise RuntimeError(f"HTTP request failed: {e}") from e
        except asyncio.TimeoutError:
            self._pending_requests.pop(self._request_id, None)
            raise RuntimeError("Request timed out") from None

    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a notification over SSE using HTTP POST."""
        if not self.client:
            raise RuntimeError("Not connected to MCP server")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }

        try:
            # Determine the correct endpoint for sending notifications
            if self._session_id:
                # Extract base server URL (remove path) and add messages endpoint
                from urllib.parse import urlparse
                parsed = urlparse(self.uri)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                messages_url = f"{base_url}/messages/?session_id={self._session_id}"
            else:
                # Fallback to the base endpoint
                messages_url = self.uri
                
            # Send notification via HTTP POST
            await self.client.post(messages_url, json=notification, timeout=10.0)
        except httpx.HTTPError as e:
            print(f"Failed to send notification: {e}")

    async def _listen(self) -> None:
        """Listen for incoming SSE events."""
        if not self.client:
            return
        try:
            async with aconnect_sse(self.client, "GET", self.uri) as sse:
                async for event in sse.aiter_sse():
                    try:
                        # Try to parse as JSON first
                        data = json.loads(event.data)
                        if "id" in data and data["id"] in self._pending_requests:
                            future = self._pending_requests.pop(data["id"])
                            if "error" in data:
                                future.set_exception(Exception(f"MCP Error: {data['error']}"))
                            else:
                                future.set_result(data.get("result", {}))
                        else:
                            # This is a server-initiated notification, we can just print it for now
                            print(f"[SSE Notification] Event: {event.event}, Data: {event.data}")
                    except json.JSONDecodeError:
                        # Handle non-JSON data, extract session ID if present
                        if event.data.startswith("/messages/?session_id="):
                            self._session_id = event.data.split("session_id=")[1]
                            print(f"[SSE] Extracted session ID: {self._session_id}")
                        else:
                            print(f"[SSE Warning] Received non-JSON data: {event.data}")
                        continue
        except httpx.ConnectError as e:
            print(f"SSE connection error: {e}")
        except Exception as e:
            print(f"An error occurred in the SSE listener: {e}")


class StreamableHttpMCPTransport(MCPTransport):
    """Streamable HTTP-based MCP transport."""

    def __init__(self, uri: str, timeout: float = 30.0):
        self.uri = uri
        self.timeout = timeout
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
            response = await self.client.post(self.uri, json=request, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP request failed: {e}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"HTTP request failed: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to decode JSON response: {e}") from e

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

    def __init__(self, command: List[str], timeout: float = 30.0):
        self.command = command
        self.timeout = timeout
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
            response = await asyncio.wait_for(future, timeout=self.timeout)
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

    def __init__(self, transport: MCPTransport, client_info: MCPClientInfo, timeout: float = 30.0):
        self.transport = transport
        self.client_info = client_info
        self.server_info: Optional[MCPServerInfo] = None
        self._tools: Dict[str, Dict[str, Any]] = {}
        self.timeout = timeout
        # Pass timeout to transport if it supports it
        if hasattr(self.transport, 'timeout'):
            self.transport.timeout = timeout

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
                from ..core.tool_results import ToolErrorInfo
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error=ToolErrorInfo(
                        code=ToolErrorCodes.EXECUTION_FAILED,
                        message=response["error"].get("message", "MCP tool execution failed"),
                        details=response
                    )
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
            from ..core.tool_results import ToolErrorInfo
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=ToolErrorInfo(
                    code=ToolErrorCodes.EXECUTION_FAILED,
                    message=f"MCP tool execution failed: {e!s}",
                    details={"error": str(e)}
                )
            )


def create_mcp_stdio_client(command: List[str], client_name: str = "JAF", client_version: str = "2.0.0", timeout: float = 30.0) -> MCPClient:
    """Create an MCP client using stdio transport."""
    transport = StdioMCPTransport(command, timeout=timeout)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info, timeout=timeout)


def create_mcp_sse_client(uri: str, client_name: str = "JAF", client_version: str = "2.0.0", timeout: float = 30.0) -> MCPClient:
    """Create an MCP client using SSE transport."""
    transport = SSEMCPTransport(uri, timeout=timeout)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info, timeout=timeout)


def create_mcp_http_client(uri: str, client_name: str = "JAF", client_version: str = "2.0.0", timeout: float = 30.0) -> MCPClient:
    """Create an MCP client using streamable HTTP transport."""
    transport = StreamableHttpMCPTransport(uri, timeout=timeout)
    client_info = MCPClientInfo(name=client_name, version=client_version)
    return MCPClient(transport, client_info, timeout=timeout)

def _validate_default_value(default_value: Any, param_type: type) -> Any:
    """
    Validate that a default value is compatible with the parameter type.
    
    Args:
        default_value: The default value from the schema
        param_type: The expected Python type
        
    Returns:
        The validated default value
        
    Raises:
        ValueError: If the default value is incompatible with the type
    """
    if default_value is None:
        return None
    
    # Basic type checking
    if param_type == str and not isinstance(default_value, str):
        try:
            return str(default_value)
        except Exception:
            raise ValueError(f"Cannot convert default value {default_value!r} to string")
    elif param_type == int and not isinstance(default_value, int):
        try:
            return int(default_value)
        except Exception:
            raise ValueError(f"Cannot convert default value {default_value!r} to integer")
    elif param_type == float and not isinstance(default_value, (int, float)):
        try:
            return float(default_value)
        except Exception:
            raise ValueError(f"Cannot convert default value {default_value!r} to float")
    elif param_type == bool and not isinstance(default_value, bool):
        # Use explicit mapping for boolean conversion
        if isinstance(default_value, str):
            val = default_value.strip().lower()
            if val in ("false", "0"):
                return False
            elif val in ("true", "1"):
                return True
            else:
                raise ValueError(f"Cannot convert default value {default_value!r} to boolean")
        elif isinstance(default_value, (int, float)):
            if default_value == 0:
                return False
            elif default_value == 1:
                return True
            else:
                raise ValueError(f"Cannot convert default value {default_value!r} to boolean")
        else:
            raise ValueError(f"Cannot convert default value {default_value!r} to boolean")
    
    return default_value

def _json_schema_to_python_type(schema: Dict[str, Any], depth: int = 0, max_depth: int = 10) -> type:
    """
    Maps JSON schema types to Python types for Pydantic model creation.
    
    This function is recursive: it calls itself to handle nested objects and arrays.
    - For arrays, it recurses into the "items" schema to determine the item type.
    - For objects, it returns Dict[str, Any] (does not create nested models).
    - For deeply nested schemas, recursion depth is limited by `max_depth`.
    If the 'type' field is missing from the schema, the function returns `Any`.
    
    Args:
        schema: JSON schema dictionary
        depth: Current recursion depth
        max_depth: Maximum allowed recursion depth
        
    Returns:
        Python type corresponding to the JSON schema
        
    Raises:
        ValueError: For unsupported schema types or excessive recursion
    """
    # Prevent infinite recursion
    if depth > max_depth:
        raise ValueError(f"Maximum recursion depth ({max_depth}) exceeded in JSON schema conversion")
    
    type_str = schema.get("type")
    if type_str == "string":
        return str
    elif type_str == "integer":
        return int
    elif type_str == "number":
        return float
    elif type_str == "boolean":
        return bool
    elif type_str == "array":
        items_schema = schema.get("items", {})
        # Recursive call for nested types, incrementing depth
        item_type = _json_schema_to_python_type(items_schema, depth=depth+1, max_depth=max_depth)
        return List[item_type]  # Works with imported List from typing
    elif type_str == "object":
        # For nested objects, we can use Dict or create another dynamic model
        # For simplicity, we'll use Dict[str, Any]
        return Dict[str, Any]  # Works with imported Dict from typing
    elif type_str is None:
        # Handle case where type is not specified
        # Log a warning when type is not specified, as this may indicate a malformed schema
        logger = logging.getLogger(__name__)
        logger.warning(f"JSON schema missing 'type' field: {schema!r}. Falling back to 'Any'.")
        return Any
    else:
        # Raise an error for unsupported or unknown schema types
        raise ValueError(f"Unsupported or unknown JSON schema type: {type_str!r} in schema: {schema}")

async def create_mcp_tools_from_client(mcp_client: MCPClient) -> List[MCPTool]:
    """Create JAF tools from all available MCP tools."""
    # Client should already be initialized
    if not mcp_client.server_info:
        await mcp_client.initialize()

    tools = []
    for tool_name in mcp_client.get_available_tools():
        tool_info = mcp_client.get_tool_info(tool_name)
        if not tool_info:
            continue

        params_schema = tool_info.get("inputSchema", {})
        properties = params_schema.get("properties", {})
        required_params = params_schema.get("required", [])

        fields = {}
        for param_name, param_schema in properties.items():
            param_type = _json_schema_to_python_type(param_schema)
            
            if param_name in required_params:
                # Required field
                fields[param_name] = (param_type, ...)
            else:
                # Optional field; only set a default if 'default' is present in the schema
                if "default" in param_schema:
                    try:
                        validated_default = _validate_default_value(param_schema["default"], param_type)
                        fields[param_name] = (Optional[param_type], validated_default)
                    except ValueError as e:
                        # Log warning but use original value - let Pydantic handle final validation
                        logger = logging.getLogger(__name__)
                        logger.warning(f"⚠️ Default value validation failed for {param_name}: {e}")
                        fields[param_name] = (Optional[param_type], param_schema["default"])
                else:
                    fields[param_name] = (Optional[param_type], None)

        # Create a dynamic Pydantic model for the arguments
        ArgsModel = create_model(
            f"{tool_name.replace('_', ' ').title().replace(' ', '')}Args",
            **fields,
            __base__=MCPToolArgs,
        )

        tool = MCPTool(mcp_client, tool_name, ArgsModel)
        tools.append(tool)

    return tools
