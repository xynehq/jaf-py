"""
Tool factory functions for improved developer experience.

This module provides factory functions for creating tools with an object-based API
that is more type-safe, extensible, and self-documenting than positional arguments.
"""

import warnings
import inspect
import json
import logging
from typing import Any, Dict, Optional, Union, Awaitable, get_type_hints, get_origin, get_args

# Optional pydantic import for validation
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

from .types import (
    FunctionToolConfig,
    Tool,
    ToolSchema,
    ToolSource,
    ToolExecuteFunction,
)
from .tool_results import ToolResult


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
    # Get the function from config
    original_func = config["execute"]

    # Validate tool configuration
    logger = logging.getLogger(__name__)

    tool_name = config["name"]
    parameters = config["parameters"]

    logger.info(f"Creating tool: {tool_name}")

    # Validate parameters schema
    if parameters is None:
        logger.error(f"Tool {tool_name}: parameters is None - LLM will receive no schema!")
        raise ValueError(f"Tool '{tool_name}' has None parameters. Provide a Pydantic model class.")

    # Check if it's a Pydantic model class
    if BaseModel is None:
        logger.warning(f"Pydantic not available for tool {tool_name} validation")
    else:
        if not (isinstance(parameters, type) and issubclass(parameters, BaseModel)):
            logger.error(
                f"Tool {tool_name}: parameters must be a Pydantic BaseModel class, got {type(parameters)}"
            )
            raise ValueError(
                f"Tool '{tool_name}' parameters must be a Pydantic BaseModel class, got {type(parameters)}"
            )

    # Validate schema generation (cached for performance)
    if not hasattr(parameters, "_schema_validated"):
        try:
            # Generate schema once to validate the model is well-formed.
            # Allow empty object schemas (no parameters) for tools that take no args.
            if hasattr(parameters, "model_json_schema"):
                _ = parameters.model_json_schema()
            elif hasattr(parameters, "schema"):
                _ = parameters.schema()
            parameters._schema_validated = True
        except Exception as e:
            logger.error(f"Tool {tool_name} schema generation failed: {e}")
            raise ValueError(f"Tool '{tool_name}' schema generation failed: {e}")

    # Create schema
    tool_schema = ToolSchema(
        name=config["name"],
        description=config["description"],
        parameters=config["parameters"],
        timeout=config.get("timeout"),
    )

    # Create a new wrapper function for this tool to avoid conflicts when multiple tools use the same base function
    async def tool_wrapper(args: Any, context: Any) -> Union[str, ToolResult]:
        """Execute the tool with given arguments and context."""
        result = original_func(args, context)

        # Handle both sync and async execute functions
        if hasattr(result, "__await__"):
            return await result
        return result

    # Add tool properties and methods to the wrapper function
    tool_wrapper.schema = tool_schema
    tool_wrapper.metadata = config.get("metadata", {})
    tool_wrapper.source = config.get("source", ToolSource.NATIVE)

    # Add execute method that calls the wrapper function
    async def execute(args: Any, context: Any) -> Union[str, ToolResult]:
        """Execute the tool with given arguments and context."""
        return await tool_wrapper(args, context)

    tool_wrapper.execute = execute

    # Add __call__ method for direct execution
    async def call_method(args: Any, context: Any) -> Union[str, ToolResult]:
        """Execute the tool with given arguments and context."""
        return await tool_wrapper.execute(args, context)

    tool_wrapper.__call__ = call_method

    return tool_wrapper


def create_function_tool_legacy(
    name: str,
    description: str,
    execute: ToolExecuteFunction,
    parameters: Any,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[ToolSource] = None,
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
        stacklevel=2,
    )
    config: FunctionToolConfig = {
        "name": name,
        "description": description,
        "execute": execute,
        "parameters": parameters,
        "metadata": metadata,
        "source": source or ToolSource.NATIVE,
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
    source: Optional[ToolSource] = None,
) -> Tool:
    """
    Create an async function-based tool using legacy positional arguments.

    **DEPRECATED**: This function is deprecated. Use `create_function_tool` with
    an object-based configuration instead for better type safety and extensibility.
    """
    warnings.warn(
        "create_async_function_tool_legacy is deprecated. Use create_function_tool with object configuration instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_function_tool_legacy(name, description, execute, parameters, metadata, source)


def _extract_docstring_info(func):
    """Extract description and parameter info from function docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return func.__name__.replace("_", " ").title(), {}

    lines = doc.strip().split("\n")
    if not lines:
        return func.__name__.replace("_", " ").title(), {}

    # First non-empty line is the description
    description = lines[0].strip()

    # Look for Args section to extract parameter descriptions
    param_descriptions = {}
    in_args_section = False

    for line in lines[1:]:
        line = line.strip()
        if line.lower().startswith("args:"):
            in_args_section = True
            continue
        elif line.lower().startswith(
            ("returns:", "return:", "raises:", "raise:", "examples:", "example:")
        ):
            in_args_section = False
            continue
        elif in_args_section and line and ":" in line:
            # Parse parameter description like "location: The location to fetch the weather for."
            param_name, param_desc = line.split(":", 1)
            param_descriptions[param_name.strip()] = param_desc.strip()

    return description, param_descriptions


def _create_parameter_schema_from_signature(func):
    """Create a parameter schema from function signature and type hints."""
    try:
        # Try to use Pydantic if available
        from pydantic import BaseModel, create_model

        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Extract parameter info, excluding 'context' parameter
        fields = {}
        for param_name, param in signature.parameters.items():
            if param_name == "context":
                continue

            param_type = type_hints.get(param_name, str)

            # Handle default values
            if param.default != inspect.Parameter.empty:
                fields[param_name] = (param_type, param.default)
            else:
                fields[param_name] = (param_type, ...)

        # Create dynamic Pydantic model
        if fields:
            return create_model(f"{func.__name__}Args", **fields)
        else:
            # Return a simple BaseModel if no parameters
            return create_model(f"{func.__name__}Args")

    except ImportError:
        # Fallback to simple dict-based schema if Pydantic not available
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            if param_name == "context":
                continue

            param_type = type_hints.get(param_name, str)

            # Convert Python types to JSON schema types
            if param_type == str:
                properties[param_name] = {"type": "string"}
            elif param_type == int:
                properties[param_name] = {"type": "integer"}
            elif param_type == float:
                properties[param_name] = {"type": "number"}
            elif param_type == bool:
                properties[param_name] = {"type": "boolean"}
            else:
                properties[param_name] = {"type": "string"}  # Default fallback

            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}


def function_tool(
    func_or_name=None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[ToolSource] = None,
    timeout: Optional[float] = None,
):
    """
    Decorator to automatically create a tool from a function.

    This decorator extracts type information from function annotations and
    docstrings to automatically create a properly configured tool by adding
    tool properties and methods directly to the function.

    Can be used with or without parameters:
    - @function_tool
    - @function_tool(name="custom", description="Custom tool")

    Args:
        func_or_name: When used as @function_tool, this is the function being decorated.
                     When used as @function_tool(...), this should be None.
        name: Optional custom name for the tool (defaults to function name)
        description: Optional custom description (defaults to docstring)
        metadata: Optional metadata for the tool
        source: Optional source tracking for the tool

    Returns:
        A Tool implementation that can be used with agents.

    Example:
        ```python
        from jaf import function_tool

        @function_tool
        async def fetch_weather(location: str, context) -> str:
            '''Fetch the weather for a given location.

            Args:
                location: The location to fetch the weather for.
            '''
            # In real life, we'd fetch the weather from a weather API
            return "sunny"
        ```
    """

    def create_tool_from_func(func):
        # Extract function information
        func_name = name or func.__name__
        func_description, param_descriptions = _extract_docstring_info(func)
        if description:
            func_description = description

        # Create parameter schema
        parameters = _create_parameter_schema_from_signature(func)

        # Store the original function
        original_func = func

        # Create schema
        tool_schema = ToolSchema(
            name=func_name, description=func_description, parameters=parameters, timeout=timeout
        )

        # Add tool properties and methods to the function
        func.schema = tool_schema
        func.metadata = metadata or {}
        func.source = source or ToolSource.NATIVE

        # Add execute method that calls the original function
        async def execute(args: Any, context: Any) -> Union[str, ToolResult]:
            """Execute the tool with given arguments and context."""
            # Check if args is a Pydantic model (from JAF engine) or individual parameters (manual call)
            if hasattr(args, "model_dump"):  # Pydantic v2
                # Unpack Pydantic model to individual parameters
                kwargs = args.model_dump()
                result = original_func(**kwargs, context=context)
            elif hasattr(args, "dict"):  # Pydantic v1
                # Unpack Pydantic model to individual parameters
                kwargs = args.dict()
                result = original_func(**kwargs, context=context)
            else:
                # Assume it's already unpacked parameters (backward compatibility)
                result = original_func(args, context)

            # Handle both sync and async execute functions
            if hasattr(result, "__await__"):
                return await result
            return result

        func.execute = execute

        # Add __call__ method that provides helpful error message
        def call_method(*args, **kwargs):
            """Provide helpful error for incorrect tool usage."""
            raise TypeError(
                f"Tool '{func_name}' should be called using 'await {func_name}.execute(args, context)' "
                f"from JAF engine, or directly as 'await {func_name}(param1, param2, ..., context)' "
                f"for manual execution. Direct tool object calls are not supported."
            )

        func.__call__ = call_method

        return func

    # If func_or_name is a callable, this means the decorator was used without parentheses: @function_tool
    if callable(func_or_name):
        return create_tool_from_func(func_or_name)

    # Otherwise, this means the decorator was used with parentheses: @function_tool(...)
    # In this case, func_or_name might be None or the name parameter
    if func_or_name is not None and name is None:
        # Handle the case where the first parameter was meant to be the name
        name = func_or_name

    # Return the decorator function
    def decorator(func):
        return create_tool_from_func(func)

    return decorator
