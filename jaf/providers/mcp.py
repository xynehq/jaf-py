"""
Functional MCP provider implementation for JAF using fastmcp.

This module provides a functional approach to integrating with external
tools and services following the MCP specification, leveraging the
fastmcp library for efficient communication.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TypeVar

import mcp.types
from pydantic import BaseModel, create_model

from fastmcp.client import Client
from fastmcp.client.transports import (
    StdioTransport,
    SSETransport,
    StreamableHttpTransport,
)

from ..core.tool_results import ToolErrorCodes, ToolResult, ToolResultStatus
from ..core.types import ToolSchema

Ctx = TypeVar('Ctx')

def _json_schema_to_python_type(schema: Dict[str, Any]) -> type:
    """Maps JSON schema types to Python types for Pydantic model creation."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List,
        "object": Dict,
    }
    
    # Handle anyOf types (union types)
    if "anyOf" in schema:
        # For anyOf, find the first non-null type
        for any_type in schema["anyOf"]:
            if any_type.get("type") != "null":
                return _json_schema_to_python_type(any_type)
        return Any
    
    # Handle integer type correctly
    if schema.get("type") == "integer":
        return int
    elif schema.get("type") == "number":
        return float
    
    return type_map.get(schema.get("type", "object"), Any)

class MCPToolArgs(BaseModel):
    """Base class for MCP tool arguments."""
    pass

class FastMCPTool:
    """A tool that proxies to a FastMCP server, managing its own session."""

    def __init__(self, transport: Any, tool_info: mcp.types.Tool, args_model: type[BaseModel], client_info: mcp.types.Implementation):
        self.transport = transport
        self.tool_name = tool_info.name
        self.args_model = args_model
        self.client_info = client_info
        self._schema = ToolSchema(
            name=tool_info.name,
            description=tool_info.description or f"MCP tool: {tool_info.name}",
            parameters=args_model
        )

    @property
    def schema(self) -> ToolSchema[MCPToolArgs]:
        return self._schema

    async def execute(self, args: MCPToolArgs, context: Ctx) -> ToolResult:
        client = Client(self.transport, client_info=self.client_info)
        try:
            async with client:
                # Only include fields that were explicitly set, not defaults
                args_dict = args.model_dump(exclude_none=True, exclude_unset=True)
                result = await client.call_tool_mcp(self.tool_name, arguments=args_dict)

                if result.isError:
                    from ..core.tool_results import ToolErrorInfo
                    error_message = "MCP tool execution failed"
                    if result.content and isinstance(result.content[0], mcp.types.TextContent):
                        error_message = result.content[0].text
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        error=ToolErrorInfo(
                            code=ToolErrorCodes.EXECUTION_FAILED,
                            message=error_message,
                            details={"mcp_error": result.structuredContent}
                        )
                    )
                
                data = result.structuredContent if result.structuredContent else str(result.content)
                
                # Create proper ToolMetadata object
                from ..core.tool_results import ToolMetadata
                tool_metadata = ToolMetadata(extra={"mcp_response": str(result)})
                
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data=str(data),
                    metadata=tool_metadata
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

async def create_tools_from_transport(transport: Any, client_info: mcp.types.Implementation) -> List[FastMCPTool]:
    """Create JAF tools from an MCP transport."""
    client = Client(transport, client_info=client_info)
    tools = []
    try:
        async with client:
            tools_list = await client.list_tools()
            for tool_info in tools_list:
                # Try both inputSchema (camelCase) and input_schema (snake_case)
                params_schema = getattr(tool_info, "inputSchema", None) or getattr(tool_info, "input_schema", {}) or {}
                properties = params_schema.get("properties", {})
                required_params = params_schema.get("required", [])

                fields = {}
                for param_name, param_schema in properties.items():
                    param_type = _json_schema_to_python_type(param_schema)
                    if param_name in required_params:
                        fields[param_name] = (param_type, ...)
                    else:
                        # Don't set default values - let them be None so exclude_unset works
                        fields[param_name] = (Optional[param_type], None)
                
                # Add juspay_meta_info to all tool schemas if not already present
                if 'juspay_meta_info' not in fields:
                    fields['juspay_meta_info'] = (Optional[Dict[str, Any]], None)

                ArgsModel = create_model(
                    f"{tool_info.name.replace('_', ' ').title().replace(' ', '')}Args",
                    **fields,
                    __base__=MCPToolArgs,
                )
                tools.append(FastMCPTool(transport, tool_info, ArgsModel, client_info))
    except Exception as e:
        logging.error(f"Failed to create MCP tools: {e}")
    return tools

async def create_mcp_stdio_tools(command: List[str], client_name: str = "JAF", client_version: str = "2.0.0") -> List[FastMCPTool]:
    transport = StdioTransport(command=command[0], args=command[1:])
    client_info = mcp.types.Implementation(name=client_name, version=client_version)
    return await create_tools_from_transport(transport, client_info)

async def create_mcp_sse_tools(uri: str, client_name: str = "JAF", client_version: str = "2.0.0") -> List[FastMCPTool]:
    transport = SSETransport(url=uri)
    client_info = mcp.types.Implementation(name=client_name, version=client_version)
    return await create_tools_from_transport(transport, client_info)

async def create_mcp_http_tools(uri: str, client_name: str = "JAF", client_version: str = "2.0.0") -> List[FastMCPTool]:
    transport = StreamableHttpTransport(url=uri)
    client_info = mcp.types.Implementation(name=client_name, version=client_version)
    return await create_tools_from_transport(transport, client_info)
