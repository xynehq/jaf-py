# Model Context Protocol (MCP) Integration

JAF provides comprehensive support for the Model Context Protocol (MCP), enabling seamless integration with external tools and services. MCP allows agents to access tools and resources from external servers through standardized protocols.

## Overview

The Model Context Protocol (MCP) is an open standard that enables secure connections between host applications (like JAF) and external data sources and tools. JAF's MCP integration provides:

- **Multiple Transport Mechanisms**: Support for stdio, SSE, and HTTP transports via FastMCP
- **Timeout Support**: Comprehensive timeout configuration for all MCP operations
- **Secure Tool Integration**: Safe execution of external tools with validation
- **Dynamic Tool Discovery**: Automatic detection and integration of MCP server tools
- **Type-Safe Operations**: Pydantic-based validation for all MCP interactions
- **Production Ready**: Robust error handling and connection management

## Transport Mechanisms

JAF supports three MCP transport mechanisms via the FastMCP library:

### 1. Stdio Transport

Best for local MCP servers running as separate processes:

```python
from jaf.providers.mcp import create_mcp_stdio_tools

# Create MCP tools from a local filesystem MCP server
mcp_tools = await create_mcp_stdio_tools(
    command=['npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'],
    client_name="JAF",
    client_version="2.0.0",
    default_timeout=30.0  # 30 second default timeout
)
```

**Use Cases:**
- Local development tools
- File system operations
- Command-line utilities
- Local database connections

### 2. Server-Sent Events (SSE) Transport

Perfect for server-to-client streaming:

```python
from jaf.providers.mcp import create_mcp_sse_tools

# Create MCP tools from an SSE MCP server
mcp_tools = await create_mcp_sse_tools(
    uri='http://localhost:8080/events',
    client_name="JAF",
    client_version="2.0.0",
    default_timeout=60.0  # 60 second timeout for SSE operations
)
```

**Use Cases:**
- Event streams
- Notifications
- Log monitoring
- Status updates

### 3. HTTP Transport

For simple request-response patterns:

```python
from jaf.providers.mcp import create_mcp_http_tools

# Create MCP tools from an HTTP MCP server
mcp_tools = await create_mcp_http_tools(
    uri='http://localhost:8080/mcp',
    client_name="JAF", 
    client_version="2.0.0",
    default_timeout=120.0  # 2 minute timeout for HTTP operations
)
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
from jaf.providers.mcp import create_mcp_stdio_tools

# Automatically create JAF tools from all available MCP tools
mcp_tools = await create_mcp_stdio_tools(
    command=['mcp-server-command'],
    client_name="JAF",
    client_version="2.0.0",
    default_timeout=45.0  # Default timeout for all tools from this server
)

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
from jaf.providers.mcp import FastMCPTool

class SecureMCPTool:
    def __init__(self, mcp_tool: FastMCPTool, allowed_paths: List[str]):
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
                from jaf.core.tool_results import ToolErrorInfo
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error=ToolErrorInfo(
                        code=ToolErrorCodes.INVALID_INPUT,
                        message=f"Path '{path}' not allowed",
                        details={"path": path, "allowed_paths": self.allowed_paths}
                    )
                )
        
        # Execute the original MCP tool
        return await self.mcp_tool.execute(args, context)

# Use secure wrapper with FastMCP tools
mcp_tools = await create_mcp_stdio_tools([...])
secure_tool = SecureMCPTool(mcp_tools[0], ['/Users', '/tmp'])
```

### Custom Field Configuration

Add custom fields to all MCP tools for extended functionality:

```python
from jaf.providers.mcp import create_mcp_stdio_tools
from typing import Dict, Any

# Add custom fields to all tools (e.g., for metadata or routing)
extra_fields = {
    "juspay_meta_info": Dict[str, Any],  # Added by default for backward compatibility
    "execution_context": str,
    "priority": int
}

mcp_tools = await create_mcp_stdio_tools(
    command=['mcp-server-command'],
    client_name="JAF",
    client_version="2.0.0",
    extra_fields=extra_fields,
    default_timeout=30.0
)

# All tools will now accept these additional fields
for tool in mcp_tools:
    print(f"Tool {tool.schema.name} accepts custom fields")
```

## Production Examples

### Filesystem Agent with MCP

Complete example of a filesystem agent using MCP:

```python
import asyncio
from jaf import Agent, run_server
from jaf.providers.mcp import create_mcp_stdio_tools
from jaf.providers.model import make_litellm_provider
from jaf.core.types import RunConfig

async def create_filesystem_agent():
    # Create tools from filesystem MCP server with appropriate timeout
    mcp_tools = await create_mcp_stdio_tools(
        command=['npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'],
        client_name="JAF",
        client_version="2.0.0",
        default_timeout=30.0  # 30 second timeout for filesystem operations
    )
    
    # Create agent with filesystem capabilities
    def instructions(state):
        return """You are a filesystem assistant with access to file operations.
        You can read, write, list, and manage files safely within allowed directories.
        Always validate paths and provide helpful feedback to users."""
    
    return Agent(
        name="FilesystemAgent",
        instructions=instructions,
        tools=mcp_tools
    )

async def main():
    # Create agent
    agent = await create_filesystem_agent()
    
    # Setup providers
    model_provider = make_litellm_provider('http://localhost:4000')
    
    # Create run config with timeout configuration
    run_config = RunConfig(
        agent_registry={"FilesystemAgent": agent},
        model_provider=model_provider,
        default_tool_timeout=45.0,  # Override default timeout for all tools
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
    # Filesystem via stdio with short timeout
    fs_tools = await create_mcp_stdio_tools(
        command=['npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'],
        client_name="JAF",
        client_version="2.0.0",
        default_timeout=15.0  # Quick filesystem operations
    )
    
    # Database via HTTP with medium timeout
    db_tools = await create_mcp_http_tools(
        uri='http://localhost:8080/database',
        client_name="JAF", 
        client_version="2.0.0",
        default_timeout=60.0  # Database operations may take longer
    )
    
    # Events via SSE with long timeout
    event_tools = await create_mcp_sse_tools(
        uri='http://localhost:8080/events',
        client_name="JAF",
        client_version="2.0.0", 
        default_timeout=120.0  # Event streaming with longer timeout
    )
    
    # Combine all tools
    all_tools = fs_tools + db_tools + event_tools
    
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
async def robust_mcp_tool_creation(command):
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            mcp_tools = await create_mcp_stdio_tools(
                command=command,
                client_name="JAF",
                client_version="2.0.0",
                default_timeout=30.0
            )
            return mcp_tools
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to create MCP tools after {max_retries} attempts: {e}")
            
            print(f"Connection attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
```

### Tool Execution Safety

MCP tools in JAF include built-in timeout support, but you can also add additional safety wrappers:

```python
import asyncio
from jaf.providers.mcp import FastMCPTool
from jaf.core.tool_results import ToolResult, ToolResultStatus, ToolErrorCodes, ToolErrorInfo

class SafeMCPTool:
    def __init__(self, mcp_tool: FastMCPTool, max_retries: int = 3):
        self.mcp_tool = mcp_tool
        self.max_retries = max_retries
        self._schema = mcp_tool.schema
    
    @property
    def schema(self):
        return self._schema
    
    async def execute(self, args, context) -> ToolResult:
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # FastMCP tools handle their own timeouts
                result = await self.mcp_tool.execute(args, context)
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Backoff
                    continue
                break
        
        # All attempts failed
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=ToolErrorInfo(
                code=ToolErrorCodes.EXECUTION_FAILED,
                message=f"Tool execution failed after {self.max_retries} attempts: {last_error}",
                details={"error": str(last_error), "attempts": self.max_retries}
            )
        )

# Create tools with built-in timeout and add retry wrapper
mcp_tools = await create_mcp_stdio_tools(
    command=['mcp-server-command'],
    default_timeout=30.0  # Built-in timeout
)
safe_tool = SafeMCPTool(mcp_tools[0], max_retries=3)
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

Test MCP tools with the new FastMCP API:

```python
import pytest
from unittest.mock import AsyncMock, patch
from jaf.providers.mcp import FastMCPTool, MCPToolArgs
from pydantic import BaseModel

class FileReadArgs(BaseModel):
    path: str

@pytest.mark.asyncio 
async def test_mcp_tool_execution():
    # Mock the transport and tool info for FastMCPTool
    mock_transport = AsyncMock()
    mock_tool_info = AsyncMock()
    mock_tool_info.name = "read_file"
    mock_tool_info.description = "Read a file"
    mock_client_info = AsyncMock()
    
    # Create tool
    tool = FastMCPTool(mock_transport, mock_tool_info, FileReadArgs, mock_client_info, timeout=30.0)
    
    # Mock the Client context manager and tool execution
    with patch('jaf.providers.mcp.Client') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful result
        mock_result = AsyncMock()
        mock_result.isError = False
        mock_result.structuredContent = "File contents"
        mock_result.content = []
        mock_client.call_tool_mcp.return_value = mock_result
        
        # Test execution
        args = FileReadArgs(path="/test/file.txt")
        result = await tool.execute(args, {})
        
        assert result.status.value == "success"
        assert "File contents" in str(result.data)
```

### Integration Testing

Test with real MCP servers:

```python
@pytest.mark.asyncio
async def test_filesystem_integration():
    # Create tools from test MCP server
    try:
        tools = await create_mcp_stdio_tools(
            command=['test-mcp-server'],
            client_name="JAF-Test", 
            client_version="2.0.0",
            default_timeout=30.0
        )
        assert len(tools) > 0
        
        # Test tool execution
        if 'list_directory' in [t.schema.name for t in tools]:
            list_tool = next(t for t in tools if t.schema.name == 'list_directory')
            
            # Create args with proper model
            args_model = list_tool.schema.parameters
            args = args_model(path='/tmp')
            
            result = await list_tool.execute(args, {})
            assert result.status.value == "success"
    
    except Exception as e:
        pytest.skip(f"MCP server not available: {e}")
```

## Troubleshooting

### Common Issues

1. **Connection Failures**
   ```python
   # Check if MCP server is running
   try:
       tools = await create_mcp_stdio_tools(
           command=['mcp-server'],
           client_name="JAF",
           client_version="2.0.0",
           default_timeout=30.0
       )
   except Exception as e:
       print(f"Tool creation failed: {e}")
       # Check server command, permissions, dependencies
   ```

2. **Tool Discovery Issues**
   ```python
   # Debug tool loading
   try:
       tools = await create_mcp_stdio_tools(['mcp-server'])
       if not tools:
           print("No tools found - check server capabilities")
       else:
           print(f"Found {len(tools)} tools: {[t.schema.name for t in tools]}")
   except Exception as e:
       print(f"Tool discovery failed: {e}")
   ```

3. **Execution Errors and Timeouts**
   ```python
   # Add detailed error logging and timeout handling
   try:
       result = await tool.execute(args, context)
       if result.status.value == "error":
           print(f"Tool execution error: {result.error.message}")
           print(f"Error details: {result.error.details}")
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
