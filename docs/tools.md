# Tools Guide

Tools are the primary way agents interact with the external world in JAF. This guide covers everything you need to know about creating, using, and managing tools in Python.

## Overview

JAF tools are Python functions decorated with `@function_tool` that implement capabilities for agents to perform actions beyond text generation. Tools can:

- Perform calculations
- Make API calls
- Query databases
- Interact with file systems
- Call external services
- Generate content

## Tool Architecture

### Modern Tool Creation with @function_tool

The recommended way to create tools uses the `@function_tool` decorator for clean, type-safe definitions:

```python
from jaf import function_tool
from typing import Optional

@function_tool
async def my_tool(param1: str, param2: int = 0, context=None) -> str:
    """Tool description for agents.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
    """
    # Tool implementation here
    return f"Processed {param1} with value {param2}"
```

### Legacy Class-Based Tools (Backward Compatibility)

For existing codebases, the class-based approach is still supported:

```python
from pydantic import BaseModel, Field
from typing import Any

class MyToolArgs(BaseModel):
    """Pydantic model defining tool parameters."""
    param1: str = Field(description="Description of parameter")
    param2: int = Field(default=0, description="Optional parameter with default")

class MyTool:
    """Tool description for agents."""
    
    @property
    def schema(self):
        """Define the tool schema."""
        return type('ToolSchema', (), {
            'name': 'my_tool',
            'description': 'What this tool does',
            'parameters': MyToolArgs
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
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$', description="Valid email address")
    
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

### Tool with External API

```python
import httpx
from jaf import function_tool
import os

@function_tool
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

### Tool with Database Access

```python
import asyncpg
from jaf import ToolResponse, ToolErrorCodes

class QueryArgs(BaseModel):
    table: str = Field(description="Table to query")
    filters: Dict[str, Any] = Field(default={}, description="Query filters")
    limit: int = Field(default=10, ge=1, le=100, description="Result limit")

class DatabaseTool:
    """Query database tables."""
    
    def __init__(self, connection_pool):
        self.pool = connection_pool
    
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'query_database',
            'description': 'Query database tables with filters',
            'parameters': QueryArgs
        })()
    
    async def execute(self, args: QueryArgs, context) -> ToolResponse:
        # Security: Validate table name (whitelist approach)
        allowed_tables = {'users', 'products', 'orders'}
        if args.table not in allowed_tables:
            return ToolResponse.validation_error(
                f"Table '{args.table}' is not allowed",
                {'allowed_tables': list(allowed_tables)}
            )
        
        try:
            async with self.pool.acquire() as conn:
                # Build safe query with parameterized conditions
                where_conditions = []
                params = []
                
                for i, (key, value) in enumerate(args.filters.items(), 1):
                    # Validate column names (basic safety)
                    if not key.isalnum():
                        return ToolResponse.validation_error(
                            f"Invalid column name: {key}",
                            {'column': key}
                        )
                    
                    where_conditions.append(f"{key} = ${i}")
                    params.append(value)
                
                where_clause = ""
                if where_conditions:
                    where_clause = f" WHERE {' AND '.join(where_conditions)}"
                
                query = f"SELECT * FROM {args.table}{where_clause} LIMIT ${len(params) + 1}"
                params.append(args.limit)
                
                rows = await conn.fetch(query, *params)
                
                results = [dict(row) for row in rows]
                
                return ToolResponse.success(
                    f"Found {len(results)} records",
                    {
                        'table': args.table,
                        'count': len(results),
                        'results': results,
                        'filters': args.filters
                    }
                )
                
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"Database query failed: {str(e)}",
                {'table': args.table, 'error': str(e)}
            )
```

## Tool Response Handling

JAF provides a standardized `ToolResponse` system for consistent error handling and result formatting.

### Response Types

```python
from jaf import ToolResponse, ToolErrorCodes

# Success response
return ToolResponse.success(
    message="Operation completed successfully",
    data={"result": "value", "metadata": "info"}
)

# Validation error (user input issue)
return ToolResponse.validation_error(
    message="Invalid input provided",
    details={"field": "value", "reason": "explanation"}
)

# Execution error (tool failure)
return ToolResponse.error(
    code=ToolErrorCodes.EXECUTION_FAILED,
    message="Tool execution failed",
    details={"error": "description", "context": "info"}
)

# Authentication error
return ToolResponse.error(
    code=ToolErrorCodes.AUTHENTICATION_FAILED,
    message="Authentication required",
    details={"required_permissions": ["read", "write"]}
)

# Rate limit error
return ToolResponse.error(
    code=ToolErrorCodes.RATE_LIMITED,
    message="Too many requests",
    details={"retry_after": 60, "limit": 100}
)
```

### Error Codes

Available error codes in `ToolErrorCodes`:

- `VALIDATION_ERROR`: Input validation failed
- `AUTHENTICATION_FAILED`: Authentication required
- `PERMISSION_DENIED`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `RATE_LIMITED`: Rate limit exceeded
- `EXECUTION_FAILED`: General execution failure
- `TIMEOUT`: Operation timed out
- `EXTERNAL_SERVICE_ERROR`: External service unavailable

## Error Handling and Security

### Input Validation

Always validate and sanitize inputs:

```python
async def execute(self, args: MyArgs, context) -> ToolResponse:
    # Validate required fields
    if not args.required_field:
        return ToolResponse.validation_error(
            "Required field missing",
            {"field": "required_field"}
        )
    
    # Validate format
    if not re.match(r'^[a-zA-Z0-9_]+$', args.identifier):
        return ToolResponse.validation_error(
            "Invalid identifier format",
            {"pattern": "alphanumeric and underscore only"}
        )
    
    # Validate ranges
    if args.count < 1 or args.count > 1000:
        return ToolResponse.validation_error(
            "Count must be between 1 and 1000",
            {"provided": args.count, "range": [1, 1000]}
        )
```

### Security Best Practices

```python
class SecureCalculatorTool:
    """Calculator with security safeguards."""
    
    async def execute(self, args: CalculateArgs, context) -> ToolResponse:
        # 1. Input sanitization
        expression = args.expression.strip()
        
        # 2. Character whitelist
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return ToolResponse.validation_error(
                "Expression contains forbidden characters",
                {"allowed": list(allowed_chars)}
            )
        
        # 3. Length limits
        if len(expression) > 200:
            return ToolResponse.validation_error(
                "Expression too long",
                {"max_length": 200, "provided_length": len(expression)}
            )
        
        # 4. Pattern detection
        dangerous_patterns = ['import', 'exec', 'eval', '__']
        if any(pattern in expression.lower() for pattern in dangerous_patterns):
            return ToolResponse.validation_error(
                "Expression contains forbidden patterns",
                {"forbidden_patterns": dangerous_patterns}
            )
        
        # 5. Safe evaluation (use ast.literal_eval or a math library)
        try:
            # Use a safe math evaluator instead of eval()
            result = safe_math_eval(expression)
            return ToolResponse.success(f"{expression} = {result}")
        except Exception as e:
            return ToolResponse.error(
                ToolErrorCodes.EXECUTION_FAILED,
                f"Calculation failed: {str(e)}"
            )
```

### Context-Based Security

Use the context parameter for authorization:

```python
async def execute(self, args: MyArgs, context) -> ToolResponse:
    # Check user permissions
    if 'admin' not in context.permissions:
        return ToolResponse.error(
            ToolErrorCodes.PERMISSION_DENIED,
            "Admin permission required",
            {"required": "admin", "provided": context.permissions}
        )
    
    # Check user-specific limits
    if context.user_id in self.rate_limited_users:
        return ToolResponse.error(
            ToolErrorCodes.RATE_LIMITED,
            "User rate limited",
            {"retry_after": 300}
        )
    
    # Proceed with execution
    return await self._execute_with_permissions(args, context)
```

## Tool Registration and Usage

### Registering Tools with Agents

```python
from jaf import Agent

# Create tool instances
calculator = CalculatorTool()
weather = WeatherTool(api_key="your-api-key")
greeter = GreetingTool()

# Create agent with tools
def instructions(state):
    return "You are a helpful assistant with access to calculation, weather, and greeting tools."

agent = Agent(
    name="UtilityAgent",
    instructions=instructions,
    tools=[calculator, weather, greeter]
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
async def execute(self, args: MyArgs, context: UserContext) -> ToolResponse:
    if not context.has_permission('read'):
        return ToolResponse.error(
            ToolErrorCodes.PERMISSION_DENIED,
            "Read permission required"
        )
```

## Testing Tools

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_greeting_tool():
    tool = GreetingTool()
    
    # Test successful execution
    args = GreetArgs(name="Alice", style="friendly")
    context = UserContext(user_id="test", permissions=["user"])
    
    result = await tool.execute(args, context)
    
    assert result.status == "success"
    assert "Alice" in result.message
    assert result.data["name"] == "Alice"

@pytest.mark.asyncio
async def test_greeting_tool_validation():
    tool = GreetingTool()
    
    # Test validation error
    args = GreetArgs(name="", style="friendly")
    context = UserContext(user_id="test", permissions=["user"])
    
    result = await tool.execute(args, context)
    
    assert result.status == "validation_error"
    assert "empty" in result.message.lower()

@pytest.mark.asyncio
async def test_weather_tool_with_mock():
    # Mock the HTTP client
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'name': 'Test City',
            'main': {'temp': 20, 'humidity': 60},
            'weather': [{'description': 'sunny'}]
        }
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        tool = WeatherTool(api_key="test-key")
        args = WeatherArgs(city="Test City")
        context = UserContext(user_id="test", permissions=["user"])
        
        result = await tool.execute(args, context)
        
        assert result.status == "success"
        assert "Test City" in result.message
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_tool_with_agent():
    from jaf import run, RunState, RunConfig, Message
    
    # Create agent with tools
    agent = Agent(
        name="TestAgent",
        instructions=lambda state: "Use the greeting tool to greet users.",
        tools=[GreetingTool()]
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
    
    # Run agent
    initial_state = RunState(
        messages=[Message(role="user", content="Please greet Alice")],
        current_agent_name="TestAgent",
        context=UserContext(user_id="test", permissions=["user"])
    )
    
    config = RunConfig(
        agent_registry={"TestAgent": agent},
        model_provider=mock_provider,
        max_turns=1
    )
    
    result = await run(initial_state, config)
    
    # Verify tool was called and result is correct
    assert result.outcome.status == "success"
    assert len(result.final_state.messages) > 1
```

## Advanced Patterns

### Tool Chaining

Tools can call other tools or return instructions for follow-up:

```python
class OrchestratorTool:
    def __init__(self, sub_tools: Dict[str, Any]):
        self.sub_tools = sub_tools
    
    async def execute(self, args: OrchestratorArgs, context) -> ToolResponse:
        results = []
        
        for step in args.steps:
            if step.tool_name not in self.sub_tools:
                return ToolResponse.validation_error(
                    f"Unknown tool: {step.tool_name}",
                    {"available_tools": list(self.sub_tools.keys())}
                )
            
            tool = self.sub_tools[step.tool_name]
            result = await tool.execute(step.args, context)
            
            if result.status != "success":
                return ToolResponse.error(
                    ToolErrorCodes.EXECUTION_FAILED,
                    f"Step {step.tool_name} failed: {result.message}",
                    {"failed_step": step.tool_name, "error": result.message}
                )
            
            results.append(result)
        
        return ToolResponse.success(
            "All steps completed successfully",
            {"step_results": [r.data for r in results]}
        )
```

### Dynamic Tool Configuration

```python
class ConfigurableTool:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_features = set(config.get('features', []))
    
    @property
    def schema(self):
        # Dynamic schema based on configuration
        parameters = {}
        
        if 'basic' in self.enabled_features:
            parameters.update({
                'basic_param': (str, Field(description="Basic parameter"))
            })
        
        if 'advanced' in self.enabled_features:
            parameters.update({
                'advanced_param': (int, Field(description="Advanced parameter"))
            })
        
        DynamicArgs = type('DynamicArgs', (BaseModel,), {
            '__annotations__': parameters
        })
        
        return type('ToolSchema', (), {
            'name': self.config['name'],
            'description': self.config['description'],
            'parameters': DynamicArgs
        })()
```

## Best Practices

1. **Always validate inputs** - Use Pydantic models and custom validators
2. **Handle errors gracefully** - Return appropriate ToolResponse objects
3. **Implement security checks** - Validate permissions and sanitize inputs
4. **Use type hints** - Leverage Python's type system for better code quality
5. **Write comprehensive tests** - Test both success and failure scenarios
6. **Document your tools** - Provide clear descriptions and examples
7. **Keep tools focused** - Each tool should have a single, well-defined purpose
8. **Use async/await** - All tools should be async for better performance
9. **Log important events** - Use structured logging for debugging and monitoring
10. **Consider rate limiting** - Implement safeguards for resource-intensive operations

## Next Steps

- Learn about [Memory System](memory-system.md) for persistent conversations
- Explore [Model Providers](model-providers.md) for LLM integration
- Check out [Examples](examples.md) for real-world tool implementations
- Read the [API Reference](api-reference.md) for complete documentation