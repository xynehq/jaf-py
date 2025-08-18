# MCP Examples

This page provides practical examples of using JAF with Model Context Protocol (MCP) servers across different transport mechanisms and use cases.

## Quick Start Examples

### Filesystem Operations with Stdio

Connect to a filesystem MCP server and perform file operations:

```python
import asyncio
from jaf import Agent, run
from jaf.providers.mcp import create_mcp_stdio_client, create_mcp_tools_from_client
from jaf.providers.model import make_litellm_provider
from jaf.core.types import RunConfig, RunState, Message, ContentRole

async def filesystem_example():
    # Connect to filesystem MCP server
    mcp_client = create_mcp_stdio_client([
        'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
    ])
    
    # Create tools from MCP server
    mcp_tools = await create_mcp_tools_from_client(mcp_client)
    
    # Create filesystem agent
    def instructions(state):
        return """You are a helpful filesystem assistant. You can:
        - List directory contents
        - Read file contents
        - Write files
        - Get file information
        
        Always be helpful and explain what you're doing."""
    
    agent = Agent(
        name="FilesystemAgent",
        instructions=instructions,
        tools=mcp_tools
    )
    
    # Setup model provider
    model_provider = make_litellm_provider('http://localhost:4000')
    
    # Create initial state
    initial_state = RunState(
        messages=[
            Message(
                role=ContentRole.USER,
                content="List the files in my Desktop directory"
            )
        ],
        current_agent_name="FilesystemAgent"
    )
    
    # Create run config
    config = RunConfig(
        agent_registry={"FilesystemAgent": agent},
        model_provider=model_provider,
        max_turns=5
    )
    
    # Run the agent
    result = await run(initial_state, config)
    
    print("Agent Response:")
    for message in result.final_state.messages:
        if message.role == ContentRole.ASSISTANT:
            print(f"Assistant: {message.content}")
    
    # Cleanup
    await mcp_client.close()

if __name__ == "__main__":
    asyncio.run(filesystem_example())
```

### WebSocket MCP Integration

Connect to a WebSocket MCP server for real-time operations:

```python
import asyncio
from jaf import Agent
from jaf.providers.mcp import create_mcp_websocket_client, MCPTool, MCPToolArgs
from pydantic import BaseModel

class DatabaseQueryArgs(MCPToolArgs):
    query: str
    parameters: dict = {}

async def websocket_database_example():
    # Connect to WebSocket MCP server
    mcp_client = create_mcp_websocket_client('ws://localhost:8080/mcp')
    await mcp_client.initialize()
    
    # Create specific tools
    query_tool = MCPTool(mcp_client, "execute_query", DatabaseQueryArgs)
    
    # Create database agent
    def instructions(state):
        return """You are a database assistant. You can execute SQL queries
        and help users interact with the database safely."""
    
    agent = Agent(
        name="DatabaseAgent",
        instructions=instructions,
        tools=[query_tool]
    )
    
    print("Database agent ready with WebSocket MCP connection")
    
    # Example usage
    try:
        # Test tool execution
        args = DatabaseQueryArgs(
            query="SELECT COUNT(*) FROM users",
            parameters={}
        )
        result = await query_tool.execute(args, {})
        print(f"Query result: {result.data}")
    
    finally:
        await mcp_client.close()

if __name__ == "__main__":
    asyncio.run(websocket_database_example())
```

## Advanced Examples

### Multi-Transport MCP Server

Create a server that uses multiple MCP transports:

```python
import asyncio
import os
from jaf import Agent, run_server
from jaf.providers.mcp import (
    create_mcp_stdio_client,
    create_mcp_websocket_client,
    create_mcp_sse_client,
    create_mcp_tools_from_client
)
from jaf.providers.model import make_litellm_provider
from jaf.core.types import RunConfig

async def create_multi_transport_server():
    """Create a server with multiple MCP transports."""
    
    # Initialize multiple MCP clients
    clients = {}
    all_tools = []
    
    try:
        # Filesystem via stdio
        print("🔌 Connecting to filesystem MCP server...")
        fs_client = create_mcp_stdio_client([
            'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
        ])
        clients['filesystem'] = fs_client
        fs_tools = await create_mcp_tools_from_client(fs_client)
        all_tools.extend(fs_tools)
        print(f"✅ Filesystem: {len(fs_tools)} tools loaded")
        
        # Database via WebSocket (if available)
        try:
            print("🔌 Connecting to database MCP server...")
            db_client = create_mcp_websocket_client('ws://localhost:8080/database')
            clients['database'] = db_client
            db_tools = await create_mcp_tools_from_client(db_client)
            all_tools.extend(db_tools)
            print(f"✅ Database: {len(db_tools)} tools loaded")
        except Exception as e:
            print(f"⚠️ Database MCP server not available: {e}")
        
        # Events via SSE (if available)
        try:
            print("🔌 Connecting to events MCP server...")
            events_client = create_mcp_sse_client('http://localhost:8080/events')
            clients['events'] = events_client
            await events_client.initialize()
            print("✅ Events: SSE connection established")
        except Exception as e:
            print(f"⚠️ Events MCP server not available: {e}")
        
        # Create comprehensive agent
        def instructions(state):
            available_transports = list(clients.keys())
            tool_count = len(all_tools)
            
            return f"""You are a comprehensive assistant with access to multiple MCP servers:

**Available Transports:** {', '.join(available_transports)}
**Total Tools:** {tool_count}

**Capabilities:**
- Filesystem operations (read, write, list files)
- Database operations (if available)
- Real-time event monitoring (if available)

**Instructions:**
- Help users with complex tasks using available tools
- Explain which transport/server you're using for each operation
- Handle errors gracefully and suggest alternatives
- Provide detailed feedback about operations performed

Always be helpful and explain your actions clearly."""
        
        agent = Agent(
            name="MultiTransportAgent",
            instructions=instructions,
            tools=all_tools
        )
        
        # Setup providers
        model_provider = make_litellm_provider(
            os.getenv('LITELLM_URL', 'http://localhost:4000'),
            os.getenv('LITELLM_API_KEY')
        )
        
        # Create run config
        run_config = RunConfig(
            agent_registry={"MultiTransportAgent": agent},
            model_provider=model_provider,
            max_turns=10
        )
        
        print(f"\n🚀 Starting multi-transport MCP server...")
        print(f"📊 Total MCP clients: {len(clients)}")
        print(f"🔧 Total tools available: {len(all_tools)}")
        
        # Start server
        await run_server(
            [agent],
            run_config,
            host="127.0.0.1",
            port=3004,
            cors=True
        )
    
    except Exception as e:
        print(f"❌ Failed to start multi-transport server: {e}")
    
    finally:
        # Cleanup all clients
        for name, client in clients.items():
            try:
                await client.close()
                print(f"🔌 Closed {name} MCP client")
            except Exception as e:
                print(f"⚠️ Error closing {name} client: {e}")

if __name__ == "__main__":
    asyncio.run(create_multi_transport_server())
```

### Secure MCP Tool Wrapper

Example of creating secure wrappers for MCP tools:

```python
import os
import re
from typing import List
from jaf.providers.mcp import MCPTool, MCPToolArgs
from jaf.core.tool_results import ToolResult, ToolResultStatus, ToolErrorCodes

class SecureFilesystemTool:
    """Secure wrapper for filesystem MCP tools."""
    
    def __init__(self, mcp_tool: MCPTool, allowed_paths: List[str], max_file_size: int = 10_000_000):
        self.mcp_tool = mcp_tool
        self.allowed_paths = [os.path.abspath(path) for path in allowed_paths]
        self.max_file_size = max_file_size
        self._schema = mcp_tool.schema
    
    @property
    def schema(self):
        return self._schema
    
    def _validate_path(self, path: str) -> tuple[bool, str]:
        """Validate if path is allowed."""
        try:
            abs_path = os.path.abspath(path)
            
            # Check if path is within allowed directories
            is_allowed = any(abs_path.startswith(allowed) for allowed in self.allowed_paths)
            if not is_allowed:
                return False, f"Path '{path}' is not within allowed directories"
            
            # Check for path traversal attempts
            if '..' in path or path.startswith('/'):
                return False, f"Path '{path}' contains invalid characters"
            
            return True, ""
        
        except Exception as e:
            return False, f"Invalid path: {e}"
    
    def _validate_filename(self, filename: str) -> tuple[bool, str]:
        """Validate filename for security."""
        # Remove dangerous characters
        safe_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
        if not safe_pattern.match(filename):
            return False, f"Filename '{filename}' contains invalid characters"
        
        # Check length
        if len(filename) > 255:
            return False, f"Filename too long: {len(filename)} characters"
        
        return True, ""
    
    async def execute(self, args, context) -> ToolResult:
        """Execute with security validation."""
        try:
            # Path validation
            if hasattr(args, 'path') and args.path:
                is_valid, error_msg = self._validate_path(args.path)
                if not is_valid:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        error_code=ToolErrorCodes.INVALID_INPUT,
                        error_message=error_msg,
                        data={"path": args.path, "allowed_paths": self.allowed_paths}
                    )
            
            # Filename validation for write operations
            if hasattr(args, 'filename') and args.filename:
                is_valid, error_msg = self._validate_filename(args.filename)
                if not is_valid:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        error_code=ToolErrorCodes.INVALID_INPUT,
                        error_message=error_msg,
                        data={"filename": args.filename}
                    )
            
            # File size validation for write operations
            if hasattr(args, 'content') and args.content:
                content_size = len(str(args.content).encode('utf-8'))
                if content_size > self.max_file_size:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        error_code=ToolErrorCodes.INVALID_INPUT,
                        error_message=f"Content too large: {content_size} bytes (max: {self.max_file_size})",
                        data={"size": content_size, "max_size": self.max_file_size}
                    )
            
            # Execute the original tool
            result = await self.mcp_tool.execute(args, context)
            
            # Add security metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata['security_validated'] = True
            result.metadata['allowed_paths'] = self.allowed_paths
            
            return result
        
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_code=ToolErrorCodes.EXECUTION_FAILED,
                error_message=f"Secure tool execution failed: {e}",
                data={"error": str(e)}
            )

# Usage example
async def create_secure_filesystem_agent():
    # Connect to MCP server
    mcp_client = create_mcp_stdio_client([
        'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
    ])
    
    # Get MCP tools
    mcp_tools = await create_mcp_tools_from_client(mcp_client)
    
    # Wrap with security
    secure_tools = []
    allowed_paths = ['/Users/username/Documents', '/tmp']
    
    for mcp_tool in mcp_tools:
        secure_tool = SecureFilesystemTool(mcp_tool, allowed_paths)
        secure_tools.append(secure_tool)
    
    # Create agent with secure tools
    def instructions(state):
        return """You are a secure filesystem assistant. You can perform file operations
        within allowed directories only. All operations are validated for security."""
    
    return Agent(
        name="SecureFilesystemAgent",
        instructions=instructions,
        tools=secure_tools
    )
```

### MCP Tool Testing Framework

Example of testing MCP tools:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from jaf.providers.mcp import MCPTool, MCPToolArgs, MCPClient
from jaf.core.tool_results import ToolResultStatus

class TestMCPArgs(MCPToolArgs):
    test_param: str

class TestMCPIntegration:
    """Test suite for MCP integration."""
    
    @pytest.fixture
    async def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = AsyncMock(spec=MCPClient)
        client.get_tool_info.return_value = {
            "name": "test_tool",
            "description": "Test tool for unit testing"
        }
        return client
    
    @pytest.fixture
    def mcp_tool(self, mock_mcp_client):
        """Create an MCP tool for testing."""
        return MCPTool(mock_mcp_client, "test_tool", TestMCPArgs)
    
    @pytest.mark.asyncio
    async def test_successful_tool_execution(self, mcp_tool, mock_mcp_client):
        """Test successful tool execution."""
        # Setup mock response
        mock_mcp_client.call_tool.return_value = {
            "content": [{"type": "text", "text": "Success!"}]
        }
        
        # Execute tool
        args = TestMCPArgs(test_param="test_value")
        result = await mcp_tool.execute(args, {})
        
        # Verify results
        assert result.status == ToolResultStatus.SUCCESS
        assert "Success!" in result.data
        mock_mcp_client.call_tool.assert_called_once_with(
            "test_tool", 
            {"test_param": "test_value"}
        )
    
    @pytest.mark.asyncio
    async def test_tool_execution_error(self, mcp_tool, mock_mcp_client):
        """Test tool execution with error."""
        # Setup mock error response
        mock_mcp_client.call_tool.return_value = {
            "error": {"message": "Tool execution failed"}
        }
        
        # Execute tool
        args = TestMCPArgs(test_param="test_value")
        result = await mcp_tool.execute(args, {})
        
        # Verify error handling
        assert result.status == ToolResultStatus.ERROR
        assert "Tool execution failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_tool_execution_exception(self, mcp_tool, mock_mcp_client):
        """Test tool execution with exception."""
        # Setup mock exception
        mock_mcp_client.call_tool.side_effect = Exception("Connection failed")
        
        # Execute tool
        args = TestMCPArgs(test_param="test_value")
        result = await mcp_tool.execute(args, {})
        
        # Verify exception handling
        assert result.status == ToolResultStatus.ERROR
        assert "Connection failed" in result.error_message

# Integration test with real MCP server
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_filesystem_mcp():
    """Integration test with real filesystem MCP server."""
    try:
        # Connect to real MCP server
        from jaf.providers.mcp import create_mcp_stdio_client, create_mcp_tools_from_client
        
        client = create_mcp_stdio_client([
            'npx', '-y', '@modelcontextprotocol/server-filesystem', '/tmp'
        ])
        
        # Test connection and tool discovery
        tools = await create_mcp_tools_from_client(client)
        assert len(tools) > 0
        
        # Test a simple operation
        list_tool = next((t for t in tools if 'list' in t.schema.name.lower()), None)
        if list_tool:
            # Create dynamic args
            class DynamicArgs(MCPToolArgs):
                class Config:
                    extra = "allow"
                
                def __init__(self, **data):
                    super().__init__()
                    for key, value in data.items():
                        setattr(self, key, value)
            
            args = DynamicArgs(path="/tmp")
            result = await list_tool.execute(args, {})
            assert result.status == ToolResultStatus.SUCCESS
        
        await client.close()
    
    except Exception as e:
        pytest.skip(f"Real MCP server not available: {e}")

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Performance Monitoring for MCP

Example of monitoring MCP tool performance:

```python
import time
import asyncio
from typing import Dict, List
from dataclasses import dataclass, field
from jaf.providers.mcp import MCPTool, MCPToolArgs
from jaf.core.tool_results import ToolResult

@dataclass
class MCPPerformanceMetrics:
    """Performance metrics for MCP tools."""
    tool_name: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    execution_times: List[float] = field(default_factory=list)

class PerformanceMonitoredMCPTool:
    """MCP tool wrapper with performance monitoring."""
    
    def __init__(self, mcp_tool: MCPTool):
        self.mcp_tool = mcp_tool
        self._schema = mcp_tool.schema
        self.metrics = MCPPerformanceMetrics(tool_name=mcp_tool.tool_name)
    
    @property
    def schema(self):
        return self._schema
    
    async def execute(self, args, context) -> ToolResult:
        """Execute with performance monitoring."""
        start_time = time.time()
        
        try:
            # Execute the original tool
            result = await self.mcp_tool.execute(args, context)
            
            # Record metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, result.status.value == "success")
            
            # Add performance metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata['execution_time'] = execution_time
            result.metadata['tool_metrics'] = self.get_metrics_summary()
            
            return result
        
        except Exception as e:
            # Record error metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, False)
            raise
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update performance metrics."""
        self.metrics.execution_count += 1
        self.metrics.total_execution_time += execution_time
        self.metrics.execution_times.append(execution_time)
        
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.error_count += 1
        
        # Calculate derived metrics
        self.metrics.average_execution_time = (
            self.metrics.total_execution_time / self.metrics.execution_count
        )
        self.metrics.success_rate = (
            self.metrics.success_count / self.metrics.execution_count * 100
        )
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary."""
        return {
            "tool_name": self.metrics.tool_name,
            "execution_count": self.metrics.execution_count,
            "average_execution_time": round(self.metrics.average_execution_time, 3),
            "success_rate": round(self.metrics.success_rate, 2),
            "total_execution_time": round(self.metrics.total_execution_time, 3)
        }
    
    def get_detailed_metrics(self) -> MCPPerformanceMetrics:
        """Get detailed performance metrics."""
        return self.metrics

class MCPPerformanceMonitor:
    """Monitor performance across multiple MCP tools."""
    
    def __init__(self):
        self.monitored_tools: Dict[str, PerformanceMonitoredMCPTool] = {}
    
    def add_tool(self, mcp_tool: MCPTool) -> PerformanceMonitoredMCPTool:
        """Add a tool for monitoring."""
        monitored_tool = PerformanceMonitoredMCPTool(mcp_tool)
        self.monitored_tools[mcp_tool.tool_name] = monitored_tool
        return monitored_tool
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        report = {
            "summary": {
                "total_tools": len(self.monitored_tools),
                "total_executions": sum(
                    tool.metrics.execution_count 
                    for tool in self.monitored_tools.values()
                ),
                "overall_success_rate": 0.0,
                "average_execution_time": 0.0
            },
            "tools": {}
        }
        
        if self.monitored_tools:
            # Calculate overall metrics
            total_executions = report["summary"]["total_executions"]
            total_successes = sum(
                tool.metrics.success_count 
                for tool in self.monitored_tools.values()
            )
            total_time = sum(
                tool.metrics.total_execution_time 
                for tool in self.monitored_tools.values()
            )
            
            if total_executions > 0:
                report["summary"]["overall_success_rate"] = round(
                    total_successes / total_executions * 100, 2
                )
                report["summary"]["average_execution_time"] = round(
                    total_time / total_executions, 3
                )
            
            # Add individual tool metrics
            for tool_name, tool in self.monitored_tools.items():
                report["tools"][tool_name] = tool.get_metrics_summary()
        
        return report
    
    def print_performance_report(self):
        """Print formatted performance report."""
        report = self.get_performance_report()
        
        print("\n" + "="*60)
        print("MCP PERFORMANCE REPORT")
        print("="*60)
        
        summary = report["summary"]
        print(f"Total Tools: {summary['total_tools']}")
        print(f"Total Executions: {summary['total_executions']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']}%")
        print(f"Average Execution Time: {summary['average_execution_time']}s")
        
        print("\nTool Performance:")
        print("-" * 60)
        
        for tool_name, metrics in report["tools"].items():
            print(f"📊 {tool_name}:")
            print(f"   Executions: {metrics['execution_count']}")
            print(f"   Success Rate: {metrics['success_rate']}%")
            print(f"   Avg Time: {metrics['average_execution_time']}s")
            print()

# Usage example
async def performance_monitoring_example():
    """Example of using performance monitoring with MCP tools."""
    from jaf.providers.mcp import create_mcp_stdio_client, create_mcp_tools_from_client
    
    # Connect to MCP server
    client = create_mcp_stdio_client([
        'npx', '-y', '@modelcontextprotocol/server-filesystem', '/tmp'
    ])
    
    # Get MCP tools
    mcp_tools = await create_mcp_tools_from_client(client)
    
    # Setup performance monitoring
    monitor = MCPPerformanceMonitor()
    monitored_tools = []
    
    for mcp_tool in mcp_tools:
        monitored_tool = monitor.add_tool(mcp_tool)
        monitored_tools.append(monitored_tool)
    
    # Simulate tool usage
    print("🔧 Running performance test...")
    
    for i in range(5):
        for tool in monitored_tools[:2]:  # Test first 2 tools
            try:
                # Create dynamic args
                class DynamicArgs(MCPToolArgs):
                    class Config:
                        extra = "allow"
                    
                    def __init__(self, **data):
                        super().__init__()
                        for key, value in data.items():
                            setattr(self, key, value)
                
                args = DynamicArgs(path="/tmp")
                await tool.execute(args, {})
                
            except Exception as e:
                print(f"Tool execution failed: {e}")
    
    # Print performance report
    monitor.print_performance_report()
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(performance_monitoring_example())
```

## Production Deployment Examples

### Docker Compose with MCP Services

Example Docker Compose setup for MCP services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  jaf-server:
    build: .
    ports:
      - "3000:3000"
    environment:
      - LITELLM_URL=http://litellm:4000
      - MCP_FILESYSTEM_PATH=/app/data
      - MCP_DATABASE_URL=postgresql://user:pass@postgres:5432/mcpdb
    volumes:
      - ./data:/app/data
    depends_on:
      - postgres
      - litellm
      - mcp-filesystem
      - mcp-database

  mcp-filesystem:
    image: node:18-alpine
    command: npx @modelcontextprotocol/server-filesystem /app/data
    volumes:
      - ./data:/app/data
    ports:
      - "8001:8001"

  mcp-database:
    build: ./mcp-database
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/mcpdb
    ports:
      - "8002:8002"
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mcpdb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

volumes:
  postgres_data:
```

### Kubernetes Deployment

Example Kubernetes deployment for MCP-enabled JAF:

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaf-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jaf-mcp-server
  template:
    metadata:
      labels:
        app: jaf-mcp-server
    spec:
      containers:
      - name: jaf-server
        image: jaf-mcp:latest
        ports:
        - containerPort: 3000
        env:
        - name: LITELLM_URL
          value: "http://litellm-service:4000"
        - name: MCP_FILESYSTEM_URL
          value: "ws://mcp-filesystem-service:8001"
        - name: MCP_DATABASE_URL
          value: "ws://mcp-database-service:8002"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"

---
apiVersion: v1
kind: Service
metadata:
  name: jaf-mcp-service
spec:
  selector:
    app: jaf-mcp-server
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

These examples demonstrate various aspects of MCP integration with JAF, from basic usage to advanced production deployments. Each example includes error handling, security considerations, and best practices for real-world usage.
