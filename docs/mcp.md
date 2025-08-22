# Model Context Protocol (MCP) Integration

JAF provides comprehensive support for the Model Context Protocol (MCP), enabling seamless integration with external tools and services. MCP allows agents to access tools and resources from external servers through standardized protocols.

## Overview

The Model Context Protocol (MCP) is an open standard that enables secure connections between host applications (like JAF) and external data sources and tools. JAF's MCP integration provides:

- **Multiple Transport Mechanisms**: Support for stdio, WebSocket, and SSE transports
- **Secure Tool Integration**: Safe execution of external tools with validation
- **Dynamic Tool Discovery**: Automatic detection and integration of MCP server tools
- **Type-Safe Operations**: Pydantic-based validation for all MCP interactions
- **Production Ready**: Robust error handling and connection management

## Transport Mechanisms

JAF supports all three MCP transport mechanisms:

### 1. Stdio Transport

Best for local MCP servers running as separate processes:

```python
from jaf.providers.mcp import create_mcp_stdio_client

# Connect to a local filesystem MCP server
mcp_client = create_mcp_stdio_client([
    'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
])

await mcp_client.initialize()
```

**Use Cases:**
- Local development tools
- File system operations
- Command-line utilities
- Local database connections

### 2. WebSocket Transport

Ideal for real-time, bidirectional communication:

```python
from jaf.providers.mcp import create_mcp_websocket_client

# Connect to a WebSocket MCP server
mcp_client = create_mcp_websocket_client('ws://localhost:8080/mcp')

await mcp_client.initialize()
```

**Use Cases:**
- Real-time data feeds
- Interactive services
- Persistent connections
- Streaming operations

### 3. Server-Sent Events (SSE) Transport

Perfect for server-to-client streaming:

```python
from jaf.providers.mcp import create_mcp_sse_client

# Connect to an SSE MCP server
mcp_client = create_mcp_sse_client('http://localhost:8080/events')

await mcp_client.initialize()
```

**Use Cases:**
- Event streams
- Notifications
- Log monitoring
- Status updates

### 4. HTTP Transport

For simple request-response patterns:

```python
from jaf.providers.mcp import create_mcp_http_client

# Connect to an HTTP MCP server
mcp_client = create_mcp_http_client('http://localhost:8080/mcp')

await mcp_client.initialize()
```

**Use Cases:**
- REST API integration
- Simple tool calls
- Stateless operations
- Web service integration

## Basic Usage

### Creating MCP Tools

Convert MCP server tools into JAF tools with timeout support:

```python
from jaf.providers.mcp import create_mcp_stdio_tools, create_mcp_sse_tools, create_mcp_http_tools

# Create MCP tools from stdio transport with default timeout
mcp_tools = await create_mcp_stdio_tools(
    command=['npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'],
    client_name="JAF",
    client_version="2.0.0",
    default_timeout=30.0  # 30 second default timeout for all tools
)

# Create MCP tools from SSE transport with custom timeout
sse_tools = await create_mcp_sse_tools(
    uri='http://localhost:8080/mcp',
    default_timeout=60.0  # 60 second timeout for SSE operations
)

# Create MCP tools from HTTP transport with longer timeout
http_tools = await create_mcp_http_tools(
    uri='http://localhost:8080/api/mcp',
    default_timeout=120.0  # 2 minute timeout for HTTP operations
)

# Use in an agent
from jaf import Agent

def agent_instructions(state):
    return "You can read files using the read_file tool."

agent = Agent(
    name="FileAgent",
    instructions=agent_instructions,
    tools=mcp_tools  # Tools automatically include timeout configuration
)
```

#### Timeout Configuration for MCP Tools

MCP tools in JAF support comprehensive timeout configuration:

```python
# Default timeout for all tools from a transport
mcp_tools = await create_mcp_stdio_tools(
    command=['mcp-server-command'],
    default_timeout=45.0  # All tools from this server default to 45 seconds
)

# Tools inherit the default timeout but can be overridden at RunConfig level
from jaf.core.types import RunConfig

config = RunConfig(
    agent_registry={'Agent': agent},
    model_provider=model_provider,
    default_tool_timeout=60.0,  # Override default for all tools
    max_turns=10
)
```

#### MCP Tool Timeout Hierarchy

MCP tools follow the same timeout resolution hierarchy as native JAF tools:

1. **Tool-specific timeout** (if defined in MCP server) - highest priority
2. **MCP transport default_timeout** - medium priority  
3. **RunConfig default_tool_timeout** - lower priority
4. **Global default (30 seconds)** - lowest priority

```python
# Example: Different timeout strategies for different MCP servers

# Fast local filesystem operations - short timeout
fs_tools = await create_mcp_stdio_tools(
    command=['npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'],
    default_timeout=15.0  # Quick filesystem operations
)

# Database operations - medium timeout
db_tools = await create_mcp_http_tools(
    uri='http://database-server:8080/mcp',
    default_timeout=60.0  # Database queries may take longer
)

# Heavy computation services - long timeout
compute_tools = await create_mcp_sse_tools(
    uri='http://compute-server:8080/events',
    default_timeout=300.0  # 5 minutes for complex computations
)
```

### Dynamic Tool Discovery

Automatically discover and integrate all available MCP tools:

```python
from jaf.providers.mcp import create_mcp_tools_from_client

# Connect to MCP server
mcp_client = create_mcp_stdio_client(['mcp-server-command'])

# Automatically create JAF tools from all available MCP tools
mcp_tools = await create_mcp_tools_from_client(mcp_client)

# Use all tools in an agent
agent = Agent(
    name="MCPAgent",
    instructions=lambda state: "You have access to various MCP tools.",
    tools=mcp_tools
)
```

## Advanced Features

### Secure Tool Wrapper

Create secure wrappers for MCP tools with validation:

```python
from jaf.core.tool_results import ToolResult, ToolResultStatus, ToolErrorCodes

class SecureMCPTool:
    def __init__(self, mcp_tool: MCPTool, allowed_paths: List[str]):
        self.mcp_tool = mcp_tool
        self.allowed_paths = allowed_paths
        self._schema = mcp_tool.schema
    
    @property
    def schema(self):
        return self._schema
    
    async def execute(self, args, context) -> ToolResult:
        # Validate paths for security
        if hasattr(args, 'path') and args.path:
            path = str(args.path)
            is_allowed = any(path.startswith(allowed) for allowed in self.allowed_paths)
            
            if not is_allowed:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error_code=ToolErrorCodes.INVALID_INPUT,
                    error_message=f"Path '{path}' not allowed",
                    data={"path": path, "allowed_paths": self.allowed_paths}
                )
        
        # Execute the original MCP tool
        return await self.mcp_tool.execute(args, context)

# Use secure wrapper
secure_tool = SecureMCPTool(mcp_tool, ['/Users', '/tmp'])
```

### Custom Transport Implementation

Create custom transport mechanisms:

```python
from jaf.providers.mcp import MCPTransport
import asyncio

class CustomMCPTransport(MCPTransport):
    def __init__(self, config):
        self.config = config
        self.connection = None
    
    async def connect(self):
        # Implement custom connection logic
        self.connection = await self._create_connection()
    
    async def disconnect(self):
        # Implement cleanup
        if self.connection:
            await self.connection.close()
    
    async def send_request(self, method: str, params: dict) -> dict:
        # Implement request sending
        return await self._send_and_receive(method, params)
    
    async def send_notification(self, method: str, params: dict):
        # Implement notification sending
        await self._send_notification(method, params)
```

## Production Examples

### Filesystem Agent with MCP

Complete example of a filesystem agent using MCP:

```python
import asyncio
from jaf import Agent, run_server
from jaf.providers.mcp import create_mcp_stdio_client, MCPTool, MCPToolArgs
from jaf.providers.model import make_litellm_provider
from jaf.core.types import RunConfig

class DynamicMCPArgs(MCPToolArgs):
    """Dynamic args that accept any parameters."""
    class Config:
        extra = "allow"
    
    def __init__(self, **data):
        super().__init__()
        for key, value in data.items():
            setattr(self, key, value)

async def create_filesystem_agent():
    # Connect to filesystem MCP server
    mcp_client = create_mcp_stdio_client([
        'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
    ])
    
    await mcp_client.initialize()
    
    # Create tools for all available MCP operations
    tools = []
    for tool_name in mcp_client.get_available_tools():
        mcp_tool = MCPTool(mcp_client, tool_name, DynamicMCPArgs)
        tools.append(mcp_tool)
    
    # Create agent with filesystem capabilities
    def instructions(state):
        return """You are a filesystem assistant with access to file operations.
        You can read, write, list, and manage files safely within allowed directories.
        Always validate paths and provide helpful feedback to users."""
    
    return Agent(
        name="FilesystemAgent",
        instructions=instructions,
        tools=tools
    )

async def main():
    # Create agent
    agent = await create_filesystem_agent()
    
    # Setup providers
    model_provider = make_litellm_provider('http://localhost:4000')
    
    # Create run config
    run_config = RunConfig(
        agent_registry={"FilesystemAgent": agent},
        model_provider=model_provider,
        max_turns=10
    )
    
    # Start server
    await run_server([agent], run_config, host="127.0.0.1", port=3003)

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Transport MCP Integration

Example using multiple MCP transports:

```python
async def create_multi_transport_agent():
    # Filesystem via stdio
    fs_client = create_mcp_stdio_client([
        'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
    ])
    
    # Database via WebSocket
    db_client = create_mcp_websocket_client('ws://localhost:8080/database')
    
    # Events via SSE
    events_client = create_mcp_sse_client('http://localhost:8080/events')
    
    # Initialize all clients
    await fs_client.initialize()
    await db_client.initialize()
    await events_client.initialize()
    
    # Create tools from all clients
    fs_tools = await create_mcp_tools_from_client(fs_client)
    db_tools = await create_mcp_tools_from_client(db_client)
    
    # Combine all tools
    all_tools = fs_tools + db_tools
    
    def instructions(state):
        return """You are a comprehensive assistant with access to:
        - Filesystem operations (read, write, list files)
        - Database operations (query, update, insert)
        - Real-time event monitoring
        
        Use these capabilities to help users with complex tasks."""
    
    return Agent(
        name="MultiTransportAgent",
        instructions=instructions,
        tools=all_tools
    )
```

## Error Handling

### Connection Management

Handle MCP connection errors gracefully:

```python
async def robust_mcp_connection(command):
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            mcp_client = create_mcp_stdio_client(command)
            await mcp_client.initialize()
            return mcp_client
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to connect after {max_retries} attempts: {e}")
            
            print(f"Connection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
```

### Tool Execution Safety

Implement safe tool execution with timeouts:

```python
import asyncio

class SafeMCPTool:
    def __init__(self, mcp_tool: MCPTool, timeout: float = 30.0):
        self.mcp_tool = mcp_tool
        self.timeout = timeout
        self._schema = mcp_tool.schema
    
    @property
    def schema(self):
        return self._schema
    
    async def execute(self, args, context) -> ToolResult:
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.mcp_tool.execute(args, context),
                timeout=self.timeout
            )
            return result
        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_code=ToolErrorCodes.TIMEOUT,
                error_message=f"Tool execution timed out after {self.timeout}s",
                data={"timeout": self.timeout}
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_code=ToolErrorCodes.EXECUTION_FAILED,
                error_message=f"Tool execution failed: {e}",
                data={"error": str(e)}
            )
```

## Best Practices

### 1. Security Considerations

Always validate inputs and restrict access:

```python
# Good: Validate file paths
def validate_path(path: str, allowed_dirs: List[str]) -> bool:
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(allowed) for allowed in allowed_dirs)

# Good: Sanitize inputs
def sanitize_filename(filename: str) -> str:
    # Remove dangerous characters
    return re.sub(r'[^\w\-_\.]', '', filename)
```

### 2. Resource Management

Properly manage MCP connections:

```python
class MCPManager:
    def __init__(self):
        self.clients = {}
    
    async def add_client(self, name: str, client: MCPClient):
        self.clients[name] = client
        await client.initialize()
    
    async def close_all(self):
        for client in self.clients.values():
            await client.close()
        self.clients.clear()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()

# Usage
async with MCPManager() as manager:
    await manager.add_client("fs", fs_client)
    await manager.add_client("db", db_client)
    # Clients automatically closed on exit
```

### 3. Performance Optimization

Cache tool schemas and reuse connections:

```python
class CachedMCPClient:
    def __init__(self, client: MCPClient):
        self.client = client
        self._tool_cache = {}
        self._schema_cache = {}
    
    async def get_tool(self, name: str) -> MCPTool:
        if name not in self._tool_cache:
            self._tool_cache[name] = MCPTool(self.client, name, DynamicMCPArgs)
        return self._tool_cache[name]
    
    def get_cached_tools(self) -> List[MCPTool]:
        return list(self._tool_cache.values())
```

## Testing MCP Integration

### Unit Testing

Test MCP tools with mock clients:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_mcp_tool_execution():
    # Mock MCP client
    mock_client = AsyncMock()
    mock_client.call_tool.return_value = {
        "content": [{"type": "text", "text": "File contents"}]
    }
    
    # Create tool
    tool = MCPTool(mock_client, "read_file", FileReadArgs)
    
    # Test execution
    args = FileReadArgs(path="/test/file.txt")
    result = await tool.execute(args, {})
    
    assert result.status == ToolResultStatus.SUCCESS
    assert "File contents" in result.data
    mock_client.call_tool.assert_called_once()
```

### Integration Testing

Test with real MCP servers:

```python
@pytest.mark.asyncio
async def test_filesystem_integration():
    # Start test MCP server
    client = create_mcp_stdio_client(['test-mcp-server'])
    await client.initialize()
    
    try:
        # Test tool discovery
        tools = await create_mcp_tools_from_client(client)
        assert len(tools) > 0
        
        # Test tool execution
        if 'list_directory' in [t.schema.name for t in tools]:
            list_tool = next(t for t in tools if t.schema.name == 'list_directory')
            result = await list_tool.execute({'path': '/tmp'}, {})
            assert result.status == ToolResultStatus.SUCCESS
    
    finally:
        await client.close()
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   ```python
   # Check if MCP server is running
   try:
       client = create_mcp_stdio_client(['mcp-server'])
       await client.initialize()
   except Exception as e:
       print(f"Connection failed: {e}")
       # Check server command, permissions, dependencies
   ```

2. **Tool Discovery Issues**
   ```python
   # Debug tool loading
   tools = client.get_available_tools()
   if not tools:
       print("No tools found - check server capabilities")
       print(f"Server info: {client.server_info}")
   ```

3. **Execution Errors**
   ```python
   # Add detailed error logging
   try:
       result = await tool.execute(args, context)
   except Exception as e:
       print(f"Tool execution failed: {e}")
       print(f"Args: {args}")
       print(f"Context: {context}")
   ```

### Debug Mode

Enable debug logging for MCP operations:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('jaf.providers.mcp')

# Add to MCP client
class DebugMCPClient(MCPClient):
    async def call_tool(self, name: str, arguments: dict) -> dict:
        logger.debug(f"Calling MCP tool: {name} with args: {arguments}")
        result = await super().call_tool(name, arguments)
        logger.debug(f"MCP tool result: {result}")
        return result
```

## Next Steps

- Explore [MCP Examples](mcp-examples.md) for practical implementations
- Learn about [MCP Transport Configuration](mcp-transports.md) for advanced setups
- Check [MCP Security](mcp-security.md) for production deployment guidelines
- Review [MCP Performance](mcp-performance.md) for optimization techniques
