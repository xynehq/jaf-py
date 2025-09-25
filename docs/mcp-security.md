# MCP Security

This guide covers security considerations when integrating with MCP (Model Context Protocol) servers.

## Overview

When using MCP tools in JAF, security is crucial since these tools can access external resources and execute operations on behalf of your agents.

## Security Principles

### 1. Tool Permissions

JAF validates MCP tools before execution:

```python
from jaf.providers.mcp import create_mcp_stdio_tools

# Create tools with security validation
tools = create_mcp_stdio_tools(
    command=["node", "server.js"],
    args=[],
    validate_schema=True,
    allow_dangerous=False  # Reject potentially dangerous operations
)
```

### 2. Input Validation

All MCP tool inputs are validated against their schemas:

```python
from jaf import create_function_tool, FunctionToolConfig
from pydantic import BaseModel, field_validator

class SecureFileArgs(BaseModel):
    path: str
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        # Prevent path traversal attacks
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid path: path traversal not allowed")
        return v

# Secure file operation tool
secure_tool = create_function_tool(FunctionToolConfig(
    name="secure_read_file",
    description="Read file with security validation",
    execute=lambda args, ctx: read_file_safely(args.path),
    parameters=SecureFileArgs,
    source="native"
))
```

### 3. Resource Isolation

MCP servers run in separate processes, providing isolation:

```python
# Each MCP server runs in its own process
stdio_tools = create_mcp_stdio_tools(
    command=["node", "--max-old-space-size=512", "server.js"],
    cwd="/safe/directory",
    env={"NODE_ENV": "production"}  # Restricted environment
)
```

### 4. Network Security

For MCP over SSE/HTTP, use HTTPS and authentication:

```python
from jaf.providers.mcp import create_mcp_sse_tools

# Secure SSE connection
sse_tools = create_mcp_sse_tools(
    url="https://secure-mcp-server.example.com/sse",
    headers={
        "Authorization": "Bearer your-secure-token",
        "X-API-Key": "your-api-key"
    }
)
```

## Security Best Practices

### Environment Configuration

```python
import os
from jaf.providers.mcp import create_mcp_stdio_tools

# Use environment variables for sensitive configuration
mcp_tools = create_mcp_stdio_tools(
    command=["node", "server.js"],
    env={
        "DATABASE_URL": os.getenv("SAFE_DB_URL"),
        "API_KEY": os.getenv("MCP_API_KEY"),
        # Remove sensitive environment variables
        "HOME": None,
        "USER": None,
    }
)
```

### Timeout and Resource Limits

```python
from jaf import Agent, run, RunConfig

agent = Agent(
    name="secure_agent",
    instructions="You are a secure agent with MCP tools",
    tools=mcp_tools
)

# Configure secure execution limits
result = run(
    agent=agent,
    messages=[{"role": "user", "content": "Process this safely"}],
    config=RunConfig(
        max_turns=10,
        max_tool_calls=5,
        timeout=30.0  # Prevent long-running operations
    )
)
```

### Tool Result Sanitization

```python
from jaf.core.tool_results import ToolResult, ToolResponse

def sanitize_tool_result(result: ToolResult) -> ToolResult:
    """Sanitize potentially sensitive tool results."""
    if result.status == "success" and result.data:
        # Remove or mask sensitive data
        sanitized_data = {
            k: v for k, v in result.data.items() 
            if k not in ['password', 'token', 'secret']
        }
        return ToolResult(
            status=result.status,
            data=sanitized_data,
            metadata=result.metadata
        )
    return result
```

## Security Checklist

### Pre-deployment

- [ ] Validate all MCP server implementations
- [ ] Review tool schemas for potential security issues
- [ ] Test with restricted environments
- [ ] Implement proper authentication for remote MCP servers
- [ ] Configure appropriate timeouts and resource limits

### Runtime Monitoring

- [ ] Monitor MCP tool execution times
- [ ] Log all tool calls and results
- [ ] Alert on suspicious patterns
- [ ] Implement circuit breakers for failing tools

### Maintenance

- [ ] Regularly update MCP server dependencies
- [ ] Review and rotate API keys
- [ ] Audit tool permissions and access patterns
- [ ] Update security policies based on new threats

## Common Security Issues

### Path Traversal

```python
# BAD - Vulnerable to path traversal
def read_file(path: str) -> str:
    return open(path).read()

# GOOD - Secure path validation
def read_file_secure(path: str) -> str:
    import os.path
    base_dir = "/safe/directory"
    full_path = os.path.join(base_dir, path)
    
    # Ensure path stays within base directory
    if not full_path.startswith(base_dir):
        raise ValueError("Path traversal attempt detected")
    
    return open(full_path).read()
```

### Command Injection

```python
# BAD - Vulnerable to command injection
def execute_command(cmd: str) -> str:
    import subprocess
    return subprocess.run(cmd, shell=True, capture_output=True).stdout

# GOOD - Use parameter arrays
def execute_safe_command(args: List[str]) -> str:
    import subprocess
    # Whitelist allowed commands
    allowed_commands = ["ls", "cat", "grep"]
    if args[0] not in allowed_commands:
        raise ValueError(f"Command {args[0]} not allowed")
    
    return subprocess.run(args, capture_output=True, text=True).stdout
```

## Related Documentation

- [MCP Integration Overview](mcp.md)
- [MCP Examples](mcp-examples.md)
- [Security Framework](security-framework.md)
- [Error Handling](error-handling.md)