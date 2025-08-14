"""
Tool factory functions for improved developer experience.

This module provides factory functions for creating tools with an object-based API
that is more type-safe, extensible, and self-documenting than positional arguments.
"""

import warnings
from typing import Any, Dict, Optional, Union, Awaitable

from .types import (
    FunctionToolConfig,
    Tool,
    ToolSchema,
    ToolSource,
    ToolExecuteFunction,
)
from .tool_results import ToolResult


class FunctionTool:
    """A tool implementation created from a function and configuration."""
    
    def __init__(self, config: FunctionToolConfig):
        """Initialize a function tool from configuration."""
        self._name = config['name']
        self._description = config['description']
        self._execute_func = config['execute']
        self._parameters = config['parameters']
        self._metadata = config.get('metadata', {})
        self._source = config.get('source', ToolSource.NATIVE)
        
        # Create schema
        self._schema = ToolSchema(
            name=self._name,
            description=self._description,
            parameters=self._parameters
        )
    
    @property
    def schema(self) -> ToolSchema:
        """Tool schema including name, description, and parameter validation."""
        return self._schema
    
    async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
        """Execute the tool with given arguments and context."""
        result = self._execute_func(args, context)
        
        # Handle both sync and async execute functions
        if hasattr(result, '__await__'):
            return await result
        return result

    async def __call__(self, args: Any, context: Any) -> Union[str, ToolResult]:
        """Execute the tool with given arguments and context."""
        return await self.execute(args, context)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Tool metadata."""
        return self._metadata.copy()
    
    @property
    def source(self) -> ToolSource:
        """Tool source."""
        return self._source


def create_function_tool(config: FunctionToolConfig) -> Tool:
    """
    Create a function-based tool using object configuration.
    
    This is the new, recommended API for creating tools that provides better
    type safety, extensibility, and self-documentation.
    
    Args:
        config: Tool configuration object with name, description, execute function,
               parameters, and optional metadata and source.
    
    Returns:
        A Tool implementation that can be used with agents.
    
    Example:
        ```python
        from pydantic import BaseModel
        from jaf import create_function_tool, ToolSource
        
        class GreetArgs(BaseModel):
            name: str
            
        async def greet_execute(args: GreetArgs, context) -> str:
            return f"Hello, {args.name}!"
        
        greet_tool = create_function_tool({
            'name': 'greet',
            'description': 'Greets a user by name',
            'execute': greet_execute,
            'parameters': GreetArgs,
            'metadata': {'category': 'social'},
            'source': ToolSource.NATIVE
        })
        ```
    """
    return FunctionTool(config)


def create_function_tool_legacy(
    name: str,
    description: str,
    execute: ToolExecuteFunction,
    parameters: Any,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[ToolSource] = None
) -> Tool:
    """
    Create a function-based tool using legacy positional arguments.
    
    **DEPRECATED**: This function is deprecated. Use `create_function_tool` with
    an object-based configuration instead for better type safety and extensibility.
    
    Args:
        name: The name of the tool
        description: A description of what the tool does
        execute: The function to execute when the tool is called
        parameters: Pydantic model or similar for parameter validation
        metadata: Optional metadata for the tool
        source: Optional source tracking for the tool
    
    Returns:
        A Tool implementation that can be used with agents.
    """
    warnings.warn(
        "create_function_tool_legacy is deprecated. Use create_function_tool with object configuration instead.",
        DeprecationWarning,
        stacklevel=2
    )
    config: FunctionToolConfig = {
        'name': name,
        'description': description,
        'execute': execute,
        'parameters': parameters,
        'metadata': metadata,
        'source': source or ToolSource.NATIVE
    }
    return create_function_tool(config)


def create_async_function_tool(config: FunctionToolConfig) -> Tool:
    """
    Create an async function-based tool using object configuration.
    
    This is a convenience function that's identical to create_function_tool
    but with a name that makes it clear the execute function should be async.
    
    Args:
        config: Tool configuration object with async execute function.
    
    Returns:
        A Tool implementation that can be used with agents.
    """
    return create_function_tool(config)


def create_async_function_tool_legacy(
    name: str,
    description: str,
    execute: ToolExecuteFunction,
    parameters: Any,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[ToolSource] = None
) -> Tool:
    """
    Create an async function-based tool using legacy positional arguments.
    
    **DEPRECATED**: This function is deprecated. Use `create_function_tool` with
    an object-based configuration instead for better type safety and extensibility.
    """
    warnings.warn(
        "create_async_function_tool_legacy is deprecated. Use create_function_tool with object configuration instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_function_tool_legacy(name, description, execute, parameters, metadata, source)
