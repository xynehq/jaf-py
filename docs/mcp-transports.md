# MCP Transport Configuration

This guide covers the configuration and advanced usage of all Model Context Protocol (MCP) transport mechanisms supported by JAF.

## Transport Overview

JAF supports four transport mechanisms for MCP communication:

1. **Stdio Transport** - Process-based communication via stdin/stdout
2. **WebSocket Transport** - Real-time bidirectional communication
3. **Server-Sent Events (SSE) Transport** - Server-to-client streaming
4. **HTTP Transport** - Request-response communication

Each transport has specific use cases, configuration options, and performance characteristics.

## Stdio Transport

### Overview

Stdio transport launches MCP servers as separate processes and communicates via stdin/stdout using JSON-RPC messages.

### Configuration

```python
from jaf.providers.mcp import create_mcp_stdio_client, StdioMCPTransport

# Basic stdio client
mcp_client = create_mcp_stdio_client([
    'npx', '-y', '@modelcontextprotocol/server-filesystem', '/Users'
])

# Advanced stdio transport configuration
transport = StdioMCPTransport([
    'python', 'my_mcp_server.py', '--config', 'production.json'
])

# Custom client with transport
from jaf.providers.mcp import MCPClient, MCPClientInfo

client_info = MCPClientInfo(name="JAF-Custom", version="2.0.0")
mcp_client = MCPClient(transport, client_info)
```

### Use Cases

- **Local Development**: Perfect for development and testing
- **File System Operations**: Filesystem MCP servers
- **Database Tools**: Local database utilities
- **Command Line Tools**: Wrapping CLI tools as MCP servers

### Best Practices

```python
import asyncio
import signal
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_stdio_client(command):
    """Context manager for stdio MCP client with proper cleanup."""
    client = None
    try:
        client = create_mcp_stdio_client(command)
        await client.initialize()
        yield client
    finally:
        if client:
            await client.close()

# Usage
async def stdio_example():
    async with managed_stdio_client(['mcp-server-command']) as client:
        tools = await create_mcp_tools_from_client(client)
        # Use tools...
```

### Error Handling

```python
async def robust_stdio_connection(command, max_retries=3):
    """Create robust stdio connection with retries."""
    for attempt in range(max_retries):
        try:
            client = create_mcp_stdio_client(command)
            await client.initialize()
            
            # Test connection
            tools = client.get_available_tools()
            if not tools:
                raise Exception("No tools available from MCP server")
            
            return client
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to connect after {max_retries} attempts: {e}")
            
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## WebSocket Transport

### Overview

WebSocket transport provides real-time, bidirectional communication with MCP servers over WebSocket connections.

### Configuration

```python
from jaf.providers.mcp import create_mcp_websocket_client, WebSocketMCPTransport

# Basic WebSocket client
mcp_client = create_mcp_websocket_client('ws://localhost:8080/mcp')

# Advanced WebSocket transport with custom configuration
import websockets

class CustomWebSocketTransport(WebSocketMCPTransport):
    def __init__(self, uri, **kwargs):
        super().__init__(uri)
        self.connect_kwargs = kwargs
    
    async def connect(self):
        """Connect with custom WebSocket options."""
        self.websocket = await websockets.connect(
            self.uri,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10,
            max_size=2**20,  # 1MB max message size
            **self.connect_kwargs
        )
        asyncio.create_task(self._listen())

# Usage with custom transport
transport = CustomWebSocketTransport(
    'ws://localhost:8080/mcp',
    extra_headers={'Authorization': 'Bearer token123'}
)
client_info = MCPClientInfo(name="JAF-WebSocket", version="2.0.0")
mcp_client = MCPClient(transport, client_info)
```

### Connection Management

```python
class WebSocketConnectionManager:
    """Manage WebSocket MCP connections with reconnection."""
    
    def __init__(self, uri, max_reconnects=5):
        self.uri = uri
        self.max_reconnects = max_reconnects
        self.client = None
        self.reconnect_count = 0
        self.is_connected = False
    
    async def connect(self):
        """Connect with automatic reconnection."""
        while self.reconnect_count < self.max_reconnects:
            try:
                self.client = create_mcp_websocket_client(self.uri)
                await self.client.initialize()
                self.is_connected = True
                self.reconnect_count = 0
                return self.client
            
            except Exception as e:
                self.reconnect_count += 1
                if self.reconnect_count >= self.max_reconnects:
                    raise Exception(f"Max reconnection attempts reached: {e}")
                
                wait_time = min(2 ** self.reconnect_count, 30)
                await asyncio.sleep(wait_time)
    
    async def disconnect(self):
        """Disconnect gracefully."""
        if self.client:
            await self.client.close()
            self.is_connected = False

# Usage
async def websocket_example():
    manager = WebSocketConnectionManager('ws://localhost:8080/mcp')
    try:
        client = await manager.connect()
        # Use client...
    finally:
        await manager.disconnect()
```

### Use Cases

- **Real-time Data**: Live data feeds and updates
- **Interactive Services**: Chat bots, interactive tools
- **Persistent Connections**: Long-running operations
- **Bidirectional Communication**: Server can push updates to client

## Server-Sent Events (SSE) Transport

### Overview

SSE transport provides server-to-client streaming for real-time updates and notifications.

### Configuration

```python
from jaf.providers.mcp import create_mcp_sse_client, SSEMCPTransport
import httpx

# Basic SSE client
mcp_client = create_mcp_sse_client('http://localhost:8080/events')

# Advanced SSE transport with custom configuration
class CustomSSETransport(SSEMCPTransport):
    def __init__(self, uri, headers=None, timeout=30):
        super().__init__(uri)
        self.headers = headers or {}
        self.timeout = timeout
    
    async def connect(self):
        """Connect with custom HTTP client configuration."""
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True
        )
        
        self.sse_connection = aconnect_sse(
            self.client, 
            "GET", 
            self.uri,
            headers=self.headers
        )
        
        asyncio.create_task(self._listen())

# Usage with authentication
transport = CustomSSETransport(
    'http://localhost:8080/events',
    headers={'Authorization': 'Bearer token123'},
    timeout=60
)
```

### Event Processing

```python
class SSEEventProcessor:
    """Process SSE events with custom handlers."""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.event_handlers = {}
    
    def register_handler(self, event_type, handler):
        """Register handler for specific event types."""
        self.event_handlers[event_type] = handler
    
    async def process_events(self):
        """Process incoming SSE events."""
        async with self.mcp_client.transport.sse_connection as sse:
            async for event in sse.aiter_sse():
                await self._handle_event(event)
    
    async def _handle_event(self, event):
        """Handle individual SSE event."""
        event_type = event.event or 'message'
        
        if event_type in self.event_handlers:
            try:
                await self.event_handlers[event_type](event)
            except Exception as e:
                print(f"Error handling {event_type} event: {e}")
        else:
            print(f"Unhandled event type: {event_type}")

# Usage
async def sse_example():
    client = create_mcp_sse_client('http://localhost:8080/events')
    await client.initialize()
    
    processor = SSEEventProcessor(client)
    
    # Register event handlers
    processor.register_handler('notification', handle_notification)
    processor.register_handler('update', handle_update)
    processor.register_handler('error', handle_error)
    
    # Process events
    await processor.process_events()

async def handle_notification(event):
    print(f"Notification: {event.data}")

async def handle_update(event):
    print(f"Update: {event.data}")

async def handle_error(event):
    print(f"Error: {event.data}")
```

### Use Cases

- **Event Streams**: Real-time notifications and updates
- **Log Monitoring**: Streaming log data
- **Status Updates**: System status and health monitoring
- **One-way Communication**: Server pushes data to client

## HTTP Transport

### Overview

HTTP transport provides simple request-response communication for stateless operations.

### Configuration

```python
from jaf.providers.mcp import create_mcp_http_client, StreamableHttpMCPTransport
import httpx

# Basic HTTP client
mcp_client = create_mcp_http_client('http://localhost:8080/mcp')

# Advanced HTTP transport with custom configuration
class CustomHTTPTransport(StreamableHttpMCPTransport):
    def __init__(self, uri, **client_kwargs):
        super().__init__(uri)
        self.client_kwargs = client_kwargs
    
    async def connect(self):
        """Connect with custom HTTP client configuration."""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            **self.client_kwargs
        )

# Usage with custom configuration
transport = CustomHTTPTransport(
    'http://localhost:8080/mcp',
    headers={'User-Agent': 'JAF-MCP-Client/2.0'},
    auth=('username', 'password')
)
```

### Request/Response Handling

```python
class HTTPMCPClient:
    """Enhanced HTTP MCP client with advanced features."""
    
    def __init__(self, base_url, **kwargs):
        self.base_url = base_url
        self.client_kwargs = kwargs
        self.client = None
        self.request_id = 0
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(**self.client_kwargs)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def call_tool_with_retry(self, tool_name, arguments, max_retries=3):
        """Call MCP tool with retry logic."""
        for attempt in range(max_retries):
            try:
                return await self._call_tool(tool_name, arguments)
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
    
    async def _call_tool(self, tool_name, arguments):
        """Make HTTP request to MCP server."""
        self.request_id += 1
        
        request_data = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/mcp",
            json=request_data,
            timeout=30.0
        )
        response.raise_for_status()
        
        return response.json()

# Usage
async def http_example():
    async with HTTPMCPClient(
        'http://localhost:8080',
        headers={'Authorization': 'Bearer token123'}
    ) as client:
        result = await client.call_tool_with_retry(
            'search_files',
            {'query': 'test.txt', 'path': '/home/user'}
        )
        print(f"Search result: {result}")
```

### Use Cases

- **REST API Integration**: Integrating with REST-based MCP servers
- **Stateless Operations**: Simple request-response patterns
- **Load Balanced Services**: Works well with load balancers
- **Caching**: Easy to implement HTTP-level caching

## Transport Comparison

| Feature | Stdio | WebSocket | SSE | HTTP |
|---------|-------|-----------|-----|------|
| **Bidirectional** | ✅ | ✅ | ❌ | ❌ |
| **Real-time** | ✅ | ✅ | ✅ | ❌ |
| **Persistent** | ✅ | ✅ | ✅ | ❌ |
| **Scalability** | Low | Medium | High | High |
| **Complexity** | Low | Medium | Low | Low |
| **Firewall Friendly** | ❌ | ⚠️ | ✅ | ✅ |
| **Load Balancer Support** | ❌ | ⚠️ | ✅ | ✅ |

## Multi-Transport Configuration

### Transport Selection Strategy

```python
from enum import Enum
from typing import Optional

class TransportType(Enum):
    STDIO = "stdio"
    WEBSOCKET = "websocket"
    SSE = "sse"
    HTTP = "http"

class MCPTransportFactory:
    """Factory for creating MCP transports based on configuration."""
    
    @staticmethod
    def create_transport(transport_type: TransportType, config: dict):
        """Create transport based on type and configuration."""
        if transport_type == TransportType.STDIO:
            return StdioMCPTransport(config['command'])
        
        elif transport_type == TransportType.WEBSOCKET:
            return WebSocketMCPTransport(config['uri'])
        
        elif transport_type == TransportType.SSE:
            return SSEMCPTransport(config['uri'])
        
        elif transport_type == TransportType.HTTP:
            return StreamableHttpMCPTransport(config['uri'])
        
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")

class AdaptiveMCPClient:
    """MCP client that can adapt transport based on conditions."""
    
    def __init__(self, transport_configs):
        self.transport_configs = transport_configs
        self.current_client = None
        self.current_transport_type = None
    
    async def connect(self, preferred_transport: Optional[TransportType] = None):
        """Connect using preferred transport or fallback."""
        transport_order = [preferred_transport] if preferred_transport else []
        transport_order.extend([
            TransportType.WEBSOCKET,
            TransportType.HTTP,
            TransportType.SSE,
            TransportType.STDIO
        ])
        
        for transport_type in transport_order:
            if transport_type not in self.transport_configs:
                continue
            
            try:
                config = self.transport_configs[transport_type]
                transport = MCPTransportFactory.create_transport(transport_type, config)
                
                client_info = MCPClientInfo(name="JAF-Adaptive", version="2.0.0")
                self.current_client = MCPClient(transport, client_info)
                
                await self.current_client.initialize()
                self.current_transport_type = transport_type
                
                print(f"Connected using {transport_type.value} transport")
                return self.current_client
            
            except Exception as e:
                print(f"Failed to connect using {transport_type.value}: {e}")
                continue
        
        raise Exception("Failed to connect using any available transport")

# Usage
async def adaptive_transport_example():
    transport_configs = {
        TransportType.WEBSOCKET: {'uri': 'ws://localhost:8080/mcp'},
        TransportType.HTTP: {'uri': 'http://localhost:8080/mcp'},
        TransportType.STDIO: {'command': ['npx', '-y', '@modelcontextprotocol/server-filesystem', '/tmp']}
    }
    
    client = AdaptiveMCPClient(transport_configs)
    mcp_client = await client.connect(preferred_transport=TransportType.WEBSOCKET)
    
    # Use client...
    tools = await create_mcp_tools_from_client(mcp_client)
    print(f"Connected with {len(tools)} tools available")
```

## Performance Optimization

### Connection Pooling

```python
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager

class MCPConnectionPool:
    """Connection pool for MCP clients."""
    
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.pools: Dict[str, List[MCPClient]] = {}
        self.active_connections: Dict[str, int] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def get_client(self, transport_config) -> MCPClient:
        """Get client from pool or create new one."""
        pool_key = self._get_pool_key(transport_config)
        
        if pool_key not in self.locks:
            self.locks[pool_key] = asyncio.Lock()
        
        async with self.locks[pool_key]:
            if pool_key not in self.pools:
                self.pools[pool_key] = []
                self.active_connections[pool_key] = 0
            
            # Try to get existing client from pool
            if self.pools[pool_key]:
                return self.pools[pool_key].pop()
            
            # Create new client if under limit
            if self.active_connections[pool_key] < self.max_connections:
                client = await self._create_client(transport_config)
                self.active_connections[pool_key] += 1
                return client
            
            # Wait for available client
            while not self.pools[pool_key]:
                await asyncio.sleep(0.1)
            
            return self.pools[pool_key].pop()
    
    async def return_client(self, client: MCPClient, transport_config):
        """Return client to pool."""
        pool_key = self._get_pool_key(transport_config)
        
        async with self.locks[pool_key]:
            self.pools[pool_key].append(client)
    
    def _get_pool_key(self, transport_config) -> str:
        """Generate pool key from transport configuration."""
        return f"{transport_config['type']}:{transport_config.get('uri', transport_config.get('command', 'unknown'))}"
    
    async def _create_client(self, transport_config) -> MCPClient:
        """Create new MCP client."""
        transport_type = TransportType(transport_config['type'])
        transport = MCPTransportFactory.create_transport(transport_type, transport_config)
        
        client_info = MCPClientInfo(name="JAF-Pooled", version="2.0.0")
        client = MCPClient(transport, client_info)
        await client.initialize()
        
        return client

# Usage with context manager
@asynccontextmanager
async def pooled_mcp_client(pool: MCPConnectionPool, transport_config):
    """Context manager for pooled MCP client."""
    client = await pool.get_client(transport_config)
    try:
        yield client
    finally:
        await pool.return_client(client, transport_config)

# Example usage
async def connection_pool_example():
    pool = MCPConnectionPool(max_connections=5)
    
    transport_config = {
        'type': 'websocket',
        'uri': 'ws://localhost:8080/mcp'
    }
    
    # Use multiple clients concurrently
    tasks = []
    for i in range(10):
        task = asyncio.create_task(use_pooled_client(pool, transport_config, i))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

async def use_pooled_client(pool, transport_config, task_id):
    async with pooled_mcp_client(pool, transport_config) as client:
        tools = client.get_available_tools()
        print(f"Task {task_id}: Using client with {len(tools)} tools")
        await asyncio.sleep(1)  # Simulate work
```

## Security Considerations

### Transport Security

```python
import ssl
from jaf.providers.mcp import WebSocketMCPTransport

class SecureWebSocketTransport(WebSocketMCPTransport):
    """Secure WebSocket transport with TLS and authentication."""
    
    def __init__(self, uri, ssl_context=None, auth_token=None):
        super().__init__(uri)
        self.ssl_context = ssl_context or self._create_ssl_context()
        self.auth_token = auth_token
    
    def _create_ssl_context(self):
        """Create secure SSL context."""
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    
    async def connect(self):
        """Connect with TLS and authentication."""
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        
        self.websocket = await websockets.connect(
            self.uri,
            ssl=self.ssl_context,
            extra_headers=headers,
            ping_interval=20,
            ping_timeout=10
        )
        
        asyncio.create_task(self._listen())

# Usage
async def secure_transport_example():
    # Create secure transport
    transport = SecureWebSocketTransport(
        'wss://secure-mcp-server.com/mcp',
        auth_token='your-jwt-token'
    )
    
    client_info = MCPClientInfo(name="JAF-Secure", version="2.0.0")
    client = MCPClient(transport, client_info)
    
    await client.initialize()
    # Use secure client...
```

This comprehensive guide covers all aspects of MCP transport configuration, from basic usage to advanced production scenarios with security, performance optimization, and error handling.
