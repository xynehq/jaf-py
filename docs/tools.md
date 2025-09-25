# Tools Guide

Tools are the primary way agents interact with the external world in JAF. This guide covers everything you need to know about creating, using, and managing tools in Python.

## Overview

JAF tools are Python functions that implement capabilities for agents to perform actions beyond text generation. Tools can:

- Perform calculations
- Make API calls
- Query databases
- Interact with file systems
- Call external services
- Generate content

## Tool Architecture

### Modern Tool Creation with create_function_tool

The recommended way to create tools uses the `create_function_tool` function with object configuration for clean, type-safe definitions:

```python
from jaf import create_function_tool, FunctionToolConfig, ToolSource
from pydantic import BaseModel
from typing import Optional

class MyToolArgs(BaseModel):
    param1: str
    param2: int = 0

async def my_tool_impl(args: MyToolArgs, context) -> str:
    """Tool implementation."""
    return f"Processed {args.param1} with value {args.param2}"

# Create the tool
my_tool = create_function_tool(FunctionToolConfig(
    name="my_tool",
    description="Tool description for agents to understand its purpose",
    execute=my_tool_impl,
    parameters=MyToolArgs,
    source=ToolSource.NATIVE
))
```

### Tool Timeouts

JAF provides comprehensive timeout support to prevent tools from running indefinitely:

```python
from jaf import function_tool

# Tool with specific timeout (10 seconds)
@function_tool(timeout=10.0)
async def quick_operation(data: str, context=None) -> str:
    """Fast operation that should complete within 10 seconds."""
    # Implementation here
    return f"Processed: {data}"

# Tool with longer timeout for heavy operations
@function_tool(timeout=300.0)  # 5 minutes
async def heavy_computation(dataset: str, context=None) -> str:
    """Heavy computation that may take up to 5 minutes."""
    # Long-running implementation here
    return f"Computed: {dataset}"
```

### Timeout Configuration Hierarchy

Timeouts are resolved using this priority order:

1. **Tool-specific timeout** (highest priority)
2. **RunConfig default_tool_timeout** 
3. **Global default (30 seconds)** (lowest priority)

```python
from jaf import create_function_tool, RunConfig, Agent

# Tool with specific timeout
quick_tool = create_function_tool({
    'name': 'quick_tool',
    'description': 'Fast operation',
    'execute': quick_operation,
    'parameters': QuickArgs,
    'timeout': 5.0  # Tool-specific: 5 seconds
})

# Tool without timeout (will use RunConfig default)
default_tool = create_function_tool({
    'name': 'default_tool', 
    'description': 'Uses config default',
    'execute': default_operation,
    'parameters': DefaultArgs
    # No timeout - will use RunConfig default
})

# RunConfig with default timeout for all tools
config = RunConfig(
    agent_registry={'Agent': agent},
    model_provider=provider,
    default_tool_timeout=60.0  # 60 seconds default for all tools
)
```

### Legacy Class-Based Tools (Backward Compatibility)

For existing codebases, the class-based approach is still supported:

```python
from pydantic import BaseModel, Field
from jaf import create_function_tool, ToolSource
from typing import Any

class MyToolArgs(BaseModel):
    """Pydantic model defining tool parameters."""
    param1: str = Field(description="Description of parameter")
    param2: int = Field(default=0, description="Optional parameter with default")

async def my_tool_execute(args: MyToolArgs, context: Any) -> str:
    """Execute the tool with given arguments and context."""
    # Tool implementation here
    return f"Processed {args.param1} with {args.param2}"

# Create tool using modern object-based API with timeout
my_tool = create_function_tool({
    'name': 'my_tool',
    'description': 'What this tool does',
    'execute': my_tool_execute,
    'parameters': MyToolArgs,
    'metadata': {'category': 'utility'},
    'source': ToolSource.NATIVE,
    'timeout': 45.0  # 45 second timeout
})
```

### Legacy Class-Based API (Backward Compatibility)

For backward compatibility, JAF also supports the traditional class-based approach:

```python
class MyTool:
    """Tool description for agents."""
    
    @property
    def schema(self):
        """Define the tool schema."""
        return type('ToolSchema', (), {
            'name': 'my_tool',
            'description': 'What this tool does',
            'parameters': MyToolArgs,
            'timeout': 30.0  # Optional timeout
        })()
    
    async def execute(self, args: MyToolArgs, context: Any) -> Any:
        """Execute the tool with given arguments and context."""
        # Tool implementation here
        pass
```

## Parameter Definition with Pydantic

JAF uses Pydantic models to define tool parameters, providing automatic validation and type safety.

### Basic Parameter Types

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
from enum import Enum

class Color(str, Enum):
    RED = "red"
    GREEN = "green" 
    BLUE = "blue"

class AdvancedToolArgs(BaseModel):
    # Required string parameter
    text: str = Field(description="Text to process")
    
    # Optional parameters with defaults
    count: int = Field(default=1, description="Number of times to repeat")
    enabled: bool = Field(default=True, description="Whether to enable processing")
    
    # Constrained parameters
    rating: int = Field(ge=1, le=10, description="Rating from 1 to 10")
    email: str = Field(pattern=r'^[^@]+@[^@]+\\.[^@]+$', description="Valid email address")
    
    # Collections
    tags: List[str] = Field(default=[], description="List of tags")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    # Enums
    color: Color = Field(default=Color.BLUE, description="Color choice")
    
    # Union types
    value: Union[str, int] = Field(description="String or integer value")
    
    # Optional fields
    optional_field: Optional[str] = Field(None, description="Optional parameter")
```

### Advanced Validation

```python
from pydantic import BaseModel, Field, validator, root_validator

class ValidatedToolArgs(BaseModel):
    expression: str = Field(description="Mathematical expression")
    precision: int = Field(default=2, ge=0, le=10, description="Decimal precision")
    
    @validator('expression')
    def validate_expression(cls, v):
        """Custom validation for expression safety."""
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in v):
            raise ValueError("Expression contains invalid characters")
        return v
    
    @root_validator
    def validate_combination(cls, values):
        """Validate parameter combinations."""
        expression = values.get('expression')
        precision = values.get('precision')
        
        if expression and '*' in expression and precision > 5:
            raise ValueError("High precision not supported for multiplication")
        
        return values
```

## Tool Implementation Patterns

### Simple Tool Example

```python
from jaf import function_tool

@function_tool
async def greet(name: str, style: str = "friendly", context=None) -> str:
    """Generate a personalized greeting.
    
    Args:
        name: Name to greet
        style: Greeting style (friendly, formal, casual)
    """
    # Input validation
    if not name.strip():
        return "Error: Name cannot be empty"
    
    # Generate greeting based on style
    if style == "formal":
        greeting = f"Good day, {name}. How may I assist you?"
    elif style == "casual":
        greeting = f"Hey {name}! What's up?"
    else:  # friendly (default)
        greeting = f"Hello, {name}! Nice to meet you."
    
    return greeting
```

### Tool with External API and Timeout

```python
import httpx
from jaf import function_tool
import os

@function_tool(timeout=30.0)  # 30 second timeout for API calls
async def get_weather(city: str, units: str = "metric", context=None) -> str:
    """Get current weather for a city.
    
    Args:
        city: City name
        units: Temperature units (metric/imperial)
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "Error: Weather API key not configured"
    
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': units
    }
    
    try:
        # HTTP client timeout (shorter than tool timeout)
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            temp = data['main']['temp']
            description = data['weather'][0]['description']
            
            return f"Weather in {city}: {temp}°{'C' if units == 'metric' else 'F'}, {description}"
            
    except httpx.TimeoutException:
        return f"Error: Weather API request timed out for {city}"
    except httpx.HTTPStatusError as e:
        return f"Error: Weather API error {e.response.status_code} for {city}"
    except Exception as e:
        return f"Error: Failed to get weather for {city}: {str(e)}"
```

### Tool with Database Access and Long Timeout

```python
import asyncpg
from jaf import function_tool
from typing import Dict, Any

@function_tool(timeout=120.0)  # 2 minute timeout for database operations
async def query_database(
    table: str,
    filters: Dict[str, Any] = None,
    limit: int = 10,
    context=None
) -> str:
    """Query database tables with filters.
    
    Args:
        table: Table to query  
        filters: Query filters (default: {})
        limit: Result limit (1-100, default: 10)
    """
    if filters is None:
        filters = {}
    
    # Validate limit
    if not (1 <= limit <= 100):
        return "Error: Limit must be between 1 and 100"
    
    # Security: Validate table name (whitelist approach)
    allowed_tables = {'users', 'products', 'orders'}
    if table not in allowed_tables:
        return f"Error: Table '{table}' is not allowed. Allowed tables: {', '.join(allowed_tables)}"
    
    try:
        # Get connection pool from context (in real implementation)
        # This would be passed through the agent context
        if not hasattr(context, 'db_pool'):
            return "Error: Database connection not available"
        
        async with context.db_pool.acquire() as conn:
            # Build safe query with parameterized conditions
            where_conditions = []
            params = []
            
            for i, (key, value) in enumerate(filters.items(), 1):
                # Validate column names (basic safety)
                if not key.replace('_', '').isalnum():
                    return f"Error: Invalid column name: {key}"
                
                where_conditions.append(f"{key} = ${i}")
                params.append(value)
            
            where_clause = ""
            if where_conditions:
                where_clause = f" WHERE {' AND '.join(where_conditions)}"
            
            query = f"SELECT * FROM {table}{where_clause} LIMIT ${len(params) + 1}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            results = [dict(row) for row in rows]
            
            return f"Found {len(results)} records in {table}: {results}"
            
    except Exception as e:
        return f"Database query failed: {str(e)}"
```

## Tool Timeout Handling

### Understanding Timeout Errors

When a tool exceeds its timeout, JAF automatically returns a structured error:

```json
{
    "error": "timeout_error",
    "message": "Tool tool_name timed out after 30.0 seconds",
    "tool_name": "tool_name",
    "timeout_seconds": 30.0
}
```

### Best Practices for Timeouts

```python
from jaf import function_tool
import asyncio

# Fast operations: 5-15 seconds
@function_tool(timeout=10.0)
async def quick_calculation(expression: str, context=None) -> str:
    """Fast mathematical calculation."""
    # Quick computation
    return f"Result: {eval(expression)}"

# Medium operations: 30-120 seconds  
@function_tool(timeout=60.0)
async def api_integration(endpoint: str, context=None) -> str:
    """API call with reasonable timeout."""
    # API call implementation
    return "API response"

# Heavy operations: 2-10 minutes
@function_tool(timeout=600.0)
async def data_processing(dataset: str, context=None) -> str:
    """Heavy data processing with long timeout."""
    # Long-running computation
    return "Processing complete"

# Operations that should never timeout: use None
@function_tool(timeout=None)
async def interactive_tool(user_input: str, context=None) -> str:
    """Interactive tool that waits for user input."""
    # This tool won't timeout (use with caution)
    return "User interaction complete"
```

## Tool Response Handling

With the `@function_tool` decorator, tools return simple strings that are automatically handled by the framework. Error handling is done through return values and exceptions.

## Error Handling and Security

### Input Validation

Always validate and sanitize inputs:

```python
@function_tool(timeout=15.0)
async def validate_input_example(
    required_field: str,
    identifier: str,
    count: int,
    context=None
) -> str:
    """Example of input validation in function tools.
    
    Args:
        required_field: Required field that cannot be empty
        identifier: Alphanumeric identifier with underscores
        count: Count value between 1 and 1000
    """
    import re
    
    # Validate required fields
    if not required_field or not required_field.strip():
        return "Error: Required field is missing or empty"
    
    # Validate format
    if not re.match(r'^[a-zA-Z0-9_]+$', identifier):
        return "Error: Invalid identifier format (alphanumeric and underscore only)"
    
    # Validate ranges
    if count < 1 or count > 1000:
        return f"Error: Count must be between 1 and 1000, got {count}"
    
    return f"Validation passed: field={required_field}, id={identifier}, count={count}"
```

### Security Best Practices

```python
@function_tool(timeout=30.0)
async def secure_calculator(expression: str, context=None) -> str:
    """Calculator with comprehensive security safeguards.
    
    Args:
        expression: Mathematical expression to evaluate safely
    """
    import ast
    import operator
    
    # 1. Input sanitization
    expression = expression.strip()
    
    # 2. Character whitelist
    allowed_chars = set('0123456789+-*/(). ')
    if not all(c in allowed_chars for c in expression):
        return f"Error: Expression contains forbidden characters. Allowed: {', '.join(sorted(allowed_chars))}"
    
    # 3. Length limits
    if len(expression) > 200:
        return f"Error: Expression too long (max 200 characters, got {len(expression)})"
    
    # 4. Pattern detection
    dangerous_patterns = ['import', 'exec', 'eval', '__']
    if any(pattern in expression.lower() for pattern in dangerous_patterns):
        return f"Error: Expression contains forbidden patterns: {dangerous_patterns}"
    
    # 5. Safe evaluation using AST parsing
    try:
        def safe_eval(node):
            """Safely evaluate AST node with limited operations."""
            safe_operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }
            
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                if type(node.op) not in safe_operators:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                left = safe_eval(node.left)
                right = safe_eval(node.right)
                return safe_operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                if type(node.op) not in safe_operators:
                    raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
                operand = safe_eval(node.operand)
                return safe_operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
        
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return f"{expression} = {result}"
        
    except Exception as e:
        return f"Calculation failed: {str(e)}"
```

### Context-Based Security

Use the context parameter for authorization:

```python
@function_tool(timeout=45.0)
async def admin_operation(operation: str, data: str, context=None) -> str:
    """Example of context-based security in function tools.
    
    Args:
        operation: Administrative operation to perform
        data: Data for the operation
    """
    # Check user permissions
    if not hasattr(context, 'permissions') or 'admin' not in context.permissions:
        required_perms = 'admin'
        provided_perms = getattr(context, 'permissions', [])
        return f"Error: Admin permission required. Required: {required_perms}, Provided: {provided_perms}"
    
    # Check user-specific limits (example rate limiting)
    rate_limited_users = {'user123', 'user456'}  # This would come from a real rate limiter
    if hasattr(context, 'user_id') and context.user_id in rate_limited_users:
        return "Error: User rate limited. Please try again in 5 minutes."
    
    # Proceed with execution
    return f"Admin operation '{operation}' executed with data: {data}"
```

## Tool Registration and Usage

### Registering Tools with Agents

```python
from jaf import Agent, function_tool, RunConfig

# Create function tools using decorators with different timeouts
@function_tool(timeout=15.0)
async def calculate(expression: str, context=None) -> str:
    """Perform safe mathematical calculations."""
    # Implementation here (see examples above)
    return f"Calculated: {expression}"

@function_tool(timeout=30.0)  # Longer timeout for API calls
async def get_weather(city: str, units: str = "metric", context=None) -> str:
    """Get current weather for a city."""
    # Implementation here (see examples above)
    return f"Weather in {city}: sunny"

@function_tool(timeout=5.0)  # Quick greeting
async def greet(name: str, style: str = "friendly", context=None) -> str:
    """Generate a personalized greeting."""
    # Implementation here (see examples above)
    return f"Hello, {name}!"

# Create agent with function tools
def instructions(state):
    return "You are a helpful assistant with access to calculation, weather, and greeting tools."

agent = Agent(
    name="UtilityAgent",
    instructions=instructions,
    tools=[calculate, get_weather, greet]
)

# Configure RunConfig with default timeout
config = RunConfig(
    agent_registry={"UtilityAgent": agent},
    model_provider=model_provider,
    default_tool_timeout=60.0,  # Default 60s for tools without specific timeout
    max_turns=10
)
```

### Context Types

Define strongly-typed contexts for better type safety:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class UserContext:
    user_id: str
    permissions: List[str]
    session_id: str
    preferences: Optional[Dict[str, Any]] = None
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions
    
    def is_admin(self) -> bool:
        return 'admin' in self.permissions

# Use in tools
@function_tool(timeout=20.0)
async def context_aware_tool(data: str, context: UserContext) -> str:
    """Example tool that uses strongly-typed context."""
    if not context.has_permission('read'):
        return "Error: Read permission required"
    
    return f"Processed data for user {context.user_id}: {data}"
```

## Testing Tools

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch

@function_tool(timeout=10.0)
async def greet(name: str, style: str = "friendly", context=None) -> str:
    """Generate a personalized greeting."""
    if not name.strip():
        return "Error: Name cannot be empty"
    
    if style == "formal":
        return f"Good day, {name}. How may I assist you?"
    elif style == "casual":
        return f"Hey {name}! What's up?"
    else:  # friendly (default)
        return f"Hello, {name}! Nice to meet you."

@pytest.mark.asyncio
async def test_greeting_tool():
    from dataclasses import dataclass
    
    @dataclass
    class UserContext:
        user_id: str
        permissions: list
    
    context = UserContext(user_id="test", permissions=["user"])
    
    # Test successful execution
    result = await greet("Alice", "friendly", context)
    
    assert "Alice" in result
    assert "Hello" in result

@pytest.mark.asyncio
async def test_greeting_tool_validation():
    from dataclasses import dataclass
    
    @dataclass
    class UserContext:
        user_id: str
        permissions: list
    
    context = UserContext(user_id="test", permissions=["user"])
    
    # Test validation error
    result = await greet("", "friendly", context)
    
    assert "Error" in result
    assert "empty" in result.lower()

@pytest.mark.asyncio
async def test_tool_timeout():
    """Test tool timeout functionality."""
    import asyncio
    
    @function_tool(timeout=1.0)  # 1 second timeout
    async def slow_tool(delay: float, context=None) -> str:
        """Tool that takes longer than timeout."""
        await asyncio.sleep(delay)
        return "Should not reach here"
    
    # This would be tested within the JAF engine context
    # The engine handles timeouts automatically
    pass

@pytest.mark.asyncio
async def test_weather_tool_with_mock():
    import httpx
    
    @function_tool(timeout=30.0)
    async def get_weather(city: str, context=None) -> str:
        """Get weather with mocked HTTP client."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://api.weather.com/weather?city={city}")
            data = response.json()
            return f"Weather in {city}: {data['weather'][0]['description']}"
    
    # Mock the HTTP client
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'weather': [{'description': 'sunny'}]
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        result = await get_weather("Test City")
        
        assert "Test City" in result
        assert "sunny" in result
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_tool_with_agent():
    from jaf import run, RunState, RunConfig, Message, Agent, function_tool
    
    # Create a greeting tool using the modern decorator
    @function_tool(timeout=10.0)
    async def greet(name: str, style: str = "friendly", context=None) -> str:
        """Generate a personalized greeting."""
        if not name.strip():
            return "Error: Name cannot be empty"
        
        if style == "formal":
            return f"Good day, {name}. How may I assist you?"
        elif style == "casual":
            return f"Hey {name}! What's up?"
        else:  # friendly (default)
            return f"Hello, {name}! Nice to meet you."
    
    # Create agent with function tool
    agent = Agent(
        name="TestAgent",
        instructions=lambda state: "Use the greeting tool to greet users.",
        tools=[greet]
    )
    
    # Mock model provider
    mock_provider = MockModelProvider([{
        'message': {
            'content': '',
            'tool_calls': [{
                'id': 'test-call',
                'type': 'function',
                'function': {
                    'name': 'greet',
                    'arguments': '{"name": "Alice", "style": "friendly"}'
                }
            }]
        }
    }])
    
    # Run agent with timeout configuration
    initial_state = RunState(
        messages=[Message(role="user", content="Please greet Alice")],
        current_agent_name="TestAgent",
        context=UserContext(user_id="test", permissions=["user"])
    )
    
    config = RunConfig(
        agent_registry={"TestAgent": agent},
        model_provider=mock_provider,
        default_tool_timeout=30.0,  # Default timeout for tools
        max_turns=1
    )
    
    result = await run(initial_state, config)
    
    # Verify tool was called and result is correct
    assert result.outcome.status == "success"
    assert len(result.final_state.messages) > 1
```

## Advanced Patterns

### Tool Chaining with Timeouts

Tools can call other tools or return instructions for follow-up:

```python
from jaf import function_tool
from typing import List, Dict, Any

@function_tool(timeout=180.0)  # 3 minutes for complex workflows
async def orchestrate_workflow(
    steps: List[Dict[str, Any]], 
    context=None
) -> str:
    """Orchestrate multiple tool calls in sequence.
    
    Args:
        steps: List of steps, each containing 'tool_name' and 'args'
    """
    # Available sub-tools registry (would be configured elsewhere)
    available_tools = {
        'calculate': calculate,
        'get_weather': get_weather,
        'greet': greet
    }
    
    results = []
    
    for i, step in enumerate(steps):
        tool_name = step.get('tool_name')
        tool_args = step.get('args', {})
        
        if not tool_name:
            return f"Error: Step {i+1} missing 'tool_name'"
        
        if tool_name not in available_tools:
            available = ', '.join(available_tools.keys())
            return f"Error: Unknown tool '{tool_name}'. Available: {available}"
        
        try:
            # Call the sub-tool
            tool_func = available_tools[tool_name]
            
            # Extract individual parameters for function call
            if tool_name == 'calculate':
                result = await tool_func(tool_args.get('expression', ''), context)
            elif tool_name == 'get_weather':
                result = await tool_func(
                    tool_args.get('city', ''), 
                    tool_args.get('units', 'metric'), 
                    context
                )
            elif tool_name == 'greet':
                result = await tool_func(
                    tool_args.get('name', ''), 
                    tool_args.get('style', 'friendly'), 
                    context
                )
            else:
                result = f"Tool {tool_name} executed"
            
            # Check for errors in result
            if result.startswith('Error:'):
                return f"Step {i+1} ({tool_name}) failed: {result}"
            
            results.append(f"Step {i+1}: {result}")
            
        except Exception as e:
            return f"Step {i+1} ({tool_name}) failed with exception: {str(e)}"
    
    return f"Workflow completed successfully:\n" + "\n".join(results)
```

### Dynamic Tool Configuration

```python
from jaf import function_tool
from typing import Dict, Any, Optional

@function_tool(timeout=60.0)
async def configurable_processor(
    data: str,
    operation: str = "basic",
    advanced_options: Optional[Dict[str, Any]] = None,
    context=None
) -> str:
    """Configurable tool that adapts behavior based on parameters.
    
    Args:
        data: Data to process
        operation: Type of operation (basic, advanced, custom)
        advanced_options: Additional options for advanced operations
    """
    if advanced_options is None:
        advanced_options = {}
    
    # Basic operations
    if operation == "basic":
        return f"Basic processing of: {data}"
    
    # Advanced operations
    elif operation == "advanced":
        multiplier = advanced_options.get('multiplier', 1)
        format_style = advanced_options.get('format', 'standard')
        
        processed = data * multiplier if isinstance(data, str) else str(data)
        
        if format_style == 'uppercase':
            processed = processed.upper()
        elif format_style == 'lowercase':
            processed = processed.lower()
        
        return f"Advanced processing: {processed}"
    
    # Custom operations
    elif operation == "custom":
        custom_logic = advanced_options.get('custom_logic', 'default')
        
        if custom_logic == 'reverse':
            return f"Custom reverse: {data[::-1]}"
        elif custom_logic == 'count':
            return f"Custom count: {len(data)} characters"
        else:
            return f"Custom default: {data}"
    
    else:
        return f"Error: Unknown operation '{operation}'. Available: basic, advanced, custom"

# Factory function for creating configured tools
def create_configured_tool(enabled_features: List[str]):
    """Create a tool with specific features enabled."""
    
    @function_tool(timeout=30.0)
    async def configured_tool(
        input_data: str,
        feature_option: str = "default",
        context=None
    ) -> str:
        """Tool configured with specific features."""
        
        if feature_option == "feature1" and "feature1" in enabled_features:
            return f"Feature 1 processing: {input_data.upper()}"
        elif feature_option == "feature2" and "feature2" in enabled_features:
            return f"Feature 2 processing: {input_data.lower()}"
        elif feature_option == "default":
            return f"Default processing: {input_data}"
        else:
            available = [f for f in enabled_features] + ["default"]
            return f"Error: Feature '{feature_option}' not available. Available: {', '.join(available)}"
    
    return configured_tool
```

## Best Practices

1. **Always validate inputs** - Use Pydantic models and custom validators
2. **Handle errors gracefully** - Return clear error messages as strings
3. **Implement security checks** - Validate permissions and sanitize inputs
4. **Use type hints** - Leverage Python's type system for better code quality
5. **Write comprehensive tests** - Test both success and failure scenarios
6. **Document your tools** - Provide clear descriptions and examples
7. **Keep tools focused** - Each tool should have a single, well-defined purpose
8. **Use async/await** - All tools should be async for better performance
9. **Log important events** - Use structured logging for debugging and monitoring
10. **Consider rate limiting** - Implement safeguards for resource-intensive operations
11. **Set appropriate timeouts** - Choose timeouts based on expected operation duration
12. **Handle timeout gracefully** - Tools should be designed to handle interruption

## Timeout Best Practices

### Choosing Appropriate Timeouts

- **Quick operations (0-15 seconds)**: Simple calculations, validations, quick API calls
- **Medium operations (15-120 seconds)**: Database queries, file I/O, standard API calls  
- **Long operations (2-10 minutes)**: Data processing, complex computations, batch operations
- **Interactive operations**: Consider using no timeout (with caution) for user interactions

### Timeout Strategy by Operation Type

```python
# Network operations - account for latency and retries
@function_tool(timeout=45.0)
async def api_call_tool(endpoint: str, context=None) -> str:
    """API calls should account for network latency."""
    pass

# Database operations - account for query complexity
@function_tool(timeout=120.0) 
async def complex_query_tool(query: str, context=None) -> str:
    """Database queries may need longer timeouts."""
    pass

# File operations - account for file size and I/O speed
@function_tool(timeout=60.0)
async def file_processing_tool(file_path: str, context=None) -> str:
    """File operations depend on size and storage speed."""
    pass

# Computation - account for algorithm complexity
@function_tool(timeout=300.0)
async def heavy_computation_tool(dataset: str, context=None) -> str:
    """Complex computations may need extended timeouts."""
    pass
```

## Next Steps

- Learn about [Memory System](memory-system.md) for persistent conversations
- Explore [Model Providers](model-providers.md) for LLM integration
- Check out [Examples](examples.md) for real-world tool implementations
- Read the [API Reference](api-reference.md) for complete documentation
- See [MCP Integration](mcp.md) for connecting external tools and services
