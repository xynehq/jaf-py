"""
Functional MCP provider implementation for JAF using fastmcp.

This module provides a functional approach to integrating with external
tools and services following the MCP specification, leveraging the
fastmcp library for efficient communication.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, TypeVar, Union

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

Ctx = TypeVar("Ctx")


def _json_schema_to_python_type(schema: Dict[str, Any]) -> type:
    """Maps JSON schema types to Python types for Pydantic model creation."""
    if not isinstance(schema, dict):
        return Any

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    # Union forms - optimized to check once
    union_keys = schema.keys() & {"anyOf", "oneOf"}
    if union_keys:
        for key in union_keys:
            union_list = schema[key]
            if isinstance(union_list, list):
                for sub in union_list:
                    if isinstance(sub, dict) and sub.get("type") != "null":
                        return _json_schema_to_python_type(sub)
            return Any

    t = schema.get("type")
    # Handle 'type' as list: e.g., ["string", "null"]
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) == 1:
            return _json_schema_to_python_type({"type": non_null[0]})
        return Any

    if t in ("integer", "number", "string", "boolean", "object", "array"):
        return type_map[t]

    # Fallback
    return Any


class MCPToolArgs(BaseModel):
    """Base class for MCP tool arguments."""

    pass


class FastMCPTool:
    """A tool that proxies to a FastMCP server, managing its own session."""

    def __init__(
        self,
        transport: Union[StdioTransport, SSETransport, StreamableHttpTransport],
        tool_info: mcp.types.Tool,
        args_model: type[BaseModel],
        client_info: mcp.types.Implementation,
        timeout: Optional[float] = None,
    ):
        self.transport = transport
        self.tool_name = tool_info.name
        self.args_model = args_model
        self.client_info = client_info
        self.timeout = timeout
        self._schema = ToolSchema(
            name=tool_info.name,
            description=tool_info.description or f"MCP tool: {tool_info.name}",
            parameters=args_model,
            timeout=timeout,
        )

    @property
    def schema(self) -> ToolSchema[MCPToolArgs]:
        return self._schema

    def _convert_simple_filters_to_flat_filter(self, simple_filters: dict) -> dict:
        """
        Converts a "simple filter" dictionary to the "FlatFilter" format required by some tools.

        Simple filters are dictionaries mapping field names to values, e.g.:
            {"status": "active", "category": ["A", "B"]}
        Each key is a field name, and each value is either a single value or a list of values to match.

        Flat filters are a more structured format with two keys:
            - "clauses": a list of filter conditions, each specifying a field, a condition (e.g., "In"), and a list of values.
            - "logic": a string expressing how to combine the clauses (e.g., "0 AND 1").

        Example output:
            {
                "clauses": [
                    {"field": "status", "condition": "In", "val": ["active"]},
                    {"field": "category", "condition": "In", "val": ["A", "B"]}
                ],
                "logic": "0 AND 1"
            }

        This conversion is needed for tools that expect filters in FlatFilter format rather than as simple key-value pairs.

        Args:
            simple_filters (dict): A dictionary of field names to values (simple filter).

        Returns:
            dict: The filter in FlatFilter format.
        """
        if not simple_filters:
            return simple_filters

        clauses = []
        for i, (field, value) in enumerate(simple_filters.items()):
            # Convert single values to lists for "In" condition
            if not isinstance(value, list):
                value = [value]

            clauses.append({"field": field, "condition": "In", "val": value})

        # Create logic string: "0 AND 1 AND 2..." for all clauses
        logic = " AND ".join(str(i) for i in range(len(clauses)))

        return {"clauses": clauses, "logic": logic}

    def _transform_arguments_for_tool(self, args_dict: dict) -> dict:
        """Transform arguments based on tool-specific requirements."""
        # Handle flatFilters transformation for tools that expect FlatFilter schema
        if (
            "flatFilters" in args_dict
            and isinstance(args_dict["flatFilters"], dict)
            and "clauses" not in args_dict["flatFilters"]
        ):
            logging.info(
                f"[JAF MCP] Converting simple flatFilters to FlatFilter format for {self.tool_name}"
            )
            args_dict["flatFilters"] = self._convert_simple_filters_to_flat_filter(
                args_dict["flatFilters"]
            )
            logging.info(f"[JAF MCP] Converted flatFilters: {args_dict['flatFilters']}")

        return args_dict

    async def execute(self, args: MCPToolArgs, context: Ctx) -> ToolResult:
        client = Client(self.transport, client_info=self.client_info)
        try:
            async with client:
                # Only include fields that were explicitly set, not defaults
                args_dict = args.model_dump(exclude_none=True, exclude_unset=True)

                # Apply tool-specific argument transformations
                args_dict = self._transform_arguments_for_tool(args_dict)

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
                            details={"mcp_error": result.structuredContent},
                        ),
                    )

                data = result.structuredContent if result.structuredContent else str(result.content)

                # Create proper ToolMetadata object
                from ..core.tool_results import ToolMetadata

                tool_metadata = ToolMetadata(extra={"mcp_response": str(result)})

                return ToolResult(
                    status=ToolResultStatus.SUCCESS, data=str(data), metadata=tool_metadata
                )

        except Exception as e:
            from ..core.tool_results import ToolErrorInfo

            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=ToolErrorInfo(
                    code=ToolErrorCodes.EXECUTION_FAILED,
                    message=f"MCP tool execution failed: {e!s}",
                    details={"error": str(e)},
                ),
            )


async def create_tools_from_transport(
    transport: Union[StdioTransport, SSETransport, StreamableHttpTransport],
    client_info: mcp.types.Implementation,
    extra_fields: Optional[Dict[str, Any]] = None,
    default_timeout: Optional[float] = None,
) -> List[FastMCPTool]:
    """Create JAF tools from an MCP transport."""
    client = Client(transport, client_info=client_info)
    tools = []
    try:
        async with client:
            tools_list = await client.list_tools()
            if tools_list is None:
                tools_list = []
            for tool_info in tools_list:
                # Support both inputSchema (camelCase) and input_schema (snake_case) for compatibility with different MCP implementations.
                camel_schema = getattr(tool_info, "inputSchema", None)
                snake_schema = getattr(tool_info, "input_schema", None)

                # Choose schema with preference for camelCase when both are non-empty dicts
                if camel_schema and snake_schema:
                    if (
                        isinstance(camel_schema, dict)
                        and isinstance(snake_schema, dict)
                        and camel_schema
                        and snake_schema
                    ):
                        logging.info(
                            f"Both 'inputSchema' and 'input_schema' are present for tool '{tool_info.name}'; preferring 'inputSchema'."
                        )
                        params_schema = camel_schema
                    elif camel_schema:
                        params_schema = camel_schema
                    else:
                        params_schema = snake_schema
                elif camel_schema:
                    params_schema = camel_schema
                elif snake_schema:
                    params_schema = snake_schema
                else:
                    params_schema = {}

                # Ensure params_schema is a dict before accessing .get
                params_schema = params_schema or {}
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

                # Add any extra fields to all tool schemas if not already present
                if extra_fields:
                    for field_name, field_type in extra_fields.items():
                        if field_name not in fields:
                            fields[field_name] = (Optional[field_type], None)

                ArgsModel = create_model(
                    f"{tool_info.name.replace('_', ' ').title().replace(' ', '')}Args",
                    **fields,
                    __base__=MCPToolArgs,
                )
                tools.append(
                    FastMCPTool(transport, tool_info, ArgsModel, client_info, default_timeout)
                )
    except Exception as e:
        logging.error(f"Failed to create MCP tools: {e}")
    return tools


async def create_mcp_stdio_tools(
    command: List[str],
    client_name: str = "JAF",
    client_version: str = "2.0.0",
    extra_fields: Optional[Dict[str, Any]] = None,
    default_timeout: Optional[float] = None,
) -> List[FastMCPTool]:
    if not command:
        raise ValueError("Command list must not be empty for MCP stdio transport.")
    # Add juspay_meta_info by default for backward compatibility
    if extra_fields is None:
        extra_fields = {"juspay_meta_info": Dict[str, Any]}
    transport = StdioTransport(command=command[0], args=command[1:])
    client_info = mcp.types.Implementation(name=client_name, version=client_version)
    return await create_tools_from_transport(transport, client_info, extra_fields, default_timeout)


async def create_mcp_sse_tools(
    uri: str,
    client_name: str = "JAF",
    client_version: str = "2.0.0",
    extra_fields: Optional[Dict[str, Any]] = None,
    default_timeout: Optional[float] = None,
) -> List[FastMCPTool]:
    # Add juspay_meta_info by default for backward compatibility
    if extra_fields is None:
        extra_fields = {"juspay_meta_info": Dict[str, Any]}
    transport = SSETransport(url=uri)
    client_info = mcp.types.Implementation(name=client_name, version=client_version)
    return await create_tools_from_transport(transport, client_info, extra_fields, default_timeout)


async def create_mcp_http_tools(
    uri: str,
    client_name: str = "JAF",
    client_version: str = "2.0.0",
    extra_fields: Optional[Dict[str, Any]] = None,
    default_timeout: Optional[float] = None,
) -> List[FastMCPTool]:
    # Add juspay_meta_info by default for backward compatibility
    if extra_fields is None:
        extra_fields = {"juspay_meta_info": Dict[str, Any]}
    transport = StreamableHttpTransport(url=uri)
    client_info = mcp.types.Implementation(name=client_name, version=client_version)
    return await create_tools_from_transport(transport, client_info, extra_fields, default_timeout)
