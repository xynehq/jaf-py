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
        
        # Validate execute function
        execute_func = config['execute']
        if isinstance(execute_func, FunctionTool):
            raise ValueError(
                f"Invalid 'execute' parameter for tool '{self._name}': received a FunctionTool object instead of a callable function. "
                f"The 'execute' parameter should be a function or async function, not a FunctionTool instance. "
                f"Did you accidentally pass a tool object instead of its execute method?"
            )
        
        if not callable(execute_func):
            raise ValueError(
                f"Invalid 'execute' parameter for tool '{self._name}': expected a callable function, got {type(execute_func).__name__}. "
                f"The 'execute' parameter should be a function or async function."
            )
        
        self._execute_func = execute_func
        self._parameters = config['parameters']
        self._metadata = config.get('metadata', {})
        self._source = config.get('source', ToolSource.NATIVE)
        
        # Create schema
        self._schema = ToolSchema(
            name=self._name,
            description=self._description,
            parameters=self._parameters
        )
    
    def __call__(self, *args, **kwargs):
        """
        Make FunctionTool callable but with helpful error message.
        This prevents the 'FunctionTool object is not callable' error and provides guidance.
        """
        raise TypeError(
            f"FunctionTool '{self._name}' object is not directly callable. "
            f"Use 'await tool.execute(args, context)' instead of 'tool(args, context)'. "
            f"The correct pattern is:\n"
            f"  result = await {self._name}.execute(args, context)\n"
            f"Not:\n"
            f"  result = await {self._name}(args, context)"
        )
    
    @property
    def schema(self) -> ToolSchema:
        """Tool schema including name, description, and parameter validation."""
        return self._schema
    
    async def execute(self, args: Any, context: Any) -> Union[str, ToolResult]:
        """Execute the tool with given arguments and context."""
        # Additional safety check to ensure _execute_func is callable
        if not callable(self._execute_func):
            raise TypeError(
                f"Tool '{self._name}' execute function is not callable. "
                f"Got {type(self._execute_func).__name__}. "
                f"This should not happen if the tool was created properly."
            )
        
        # Check if execute function is a FunctionTool (this should never happen but let's be safe)
        if isinstance(self._execute_func, FunctionTool):
            raise TypeError(
                f"Tool '{self._name}' execute function is a FunctionTool object, but it should be a callable function. "
                f"This indicates a bug in tool creation where a FunctionTool was passed as the execute parameter instead of a function."
            )
        
        try:
            result = self._execute_func(args, context)
            
            # Handle both sync and async execute functions
            if hasattr(result, '__await__'):
                return await result
            return result
        except TypeError as e:
            if "object is not callable" in str(e):
                raise TypeError(
                    f"Failed to execute tool '{self._name}': {str(e)}. "
                    f"The execute function appears to not be callable. "
                    f"Execute function type: {type(self._execute_func).__name__}. "
                    f"This might indicate that a non-function object was passed as the execute parameter during tool creation."
                ) from e
            raise
    
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
    # Validate that execute is callable and not a FunctionTool object
    execute_func = config['execute']
    if isinstance(execute_func, FunctionTool):
        raise ValueError(
            f"Invalid 'execute' parameter: received a FunctionTool object instead of a callable function. "
            f"The 'execute' parameter should be a function or async function, not a FunctionTool instance. "
            f"Did you accidentally pass a tool object instead of its execute method?"
        )
    
    if not callable(execute_func):
        raise ValueError(
            f"Invalid 'execute' parameter: expected a callable function, got {type(execute_func).__name__}. "
            f"The 'execute' parameter should be a function or async function."
        )
    
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
    
    # Validate execute function
    if isinstance(execute, FunctionTool):
        raise ValueError(
            f"Invalid 'execute' parameter for tool '{name}': received a FunctionTool object instead of a callable function. "
            f"The 'execute' parameter should be a function or async function, not a FunctionTool instance. "
            f"Did you accidentally pass a tool object instead of its execute method?"
        )
    
    if not callable(execute):
        raise ValueError(
            f"Invalid 'execute' parameter for tool '{name}': expected a callable function, got {type(execute).__name__}. "
            f"The 'execute' parameter should be a function or async function."
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
