# MCP Performance

This guide covers performance considerations and optimization strategies for MCP (Model Context Protocol) integration in JAF.

## Overview

MCP tools can introduce latency and resource overhead. This guide provides strategies to optimize MCP performance while maintaining functionality.

## Performance Characteristics

### Latency Sources

1. **Process Startup**: STDIO MCP servers require process creation
2. **IPC Overhead**: Communication between JAF and MCP server
3. **Network Latency**: For remote MCP servers (SSE/HTTP)
4. **Tool Execution**: The actual tool operation time

### Benchmarking MCP Tools

```python
import time
from jaf import Agent, run
from jaf.providers.mcp import create_mcp_stdio_tools

# Benchmark MCP tool performance
def benchmark_mcp_tools():
    tools = create_mcp_stdio_tools(
        command=["node", "benchmark-server.js"]
    )
    
    agent = Agent(
        name="benchmark_agent",
        instructions="Execute the test tool",
        tools=tools
    )
    
    start_time = time.time()
    result = run(
        agent=agent,
        messages=[{"role": "user", "content": "Run performance test"}]
    )
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.2f}s")
    return result
```

## Optimization Strategies

### 1. Connection Pooling

Keep MCP connections alive to avoid startup overhead:

```python
from jaf.providers.mcp import create_mcp_stdio_tools
import atexit

class MCPConnectionPool:
    def __init__(self):
        self._connections = {}
    
    def get_tools(self, server_config):
        key = f"{server_config['command']}:{server_config.get('cwd', '')}"
        
        if key not in self._connections:
            self._connections[key] = create_mcp_stdio_tools(**server_config)
            
        return self._connections[key]
    
    def cleanup(self):
        for tools in self._connections.values():
            # Cleanup connections
            if hasattr(tools, 'cleanup'):
                tools.cleanup()

# Global connection pool
mcp_pool = MCPConnectionPool()
atexit.register(mcp_pool.cleanup)

# Use pooled connections
fast_tools = mcp_pool.get_tools({
    "command": ["node", "server.js"],
    "cwd": "/path/to/server"
})
```

### 2. Caching Tool Results

Cache expensive tool operations:

```python
from functools import lru_cache
from jaf import create_function_tool, FunctionToolConfig
from typing import Any, Dict

class CachedMCPTool:
    def __init__(self, original_tool):
        self.original_tool = original_tool
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def cached_execute(self, args_hash: str, args: Any, context: Any):
        return self.original_tool.execute(args, context)
    
    def execute(self, args: Any, context: Any):
        # Create hashable key from args
        args_hash = str(sorted(args.dict().items()) if hasattr(args, 'dict') else str(args))
        return self.cached_execute(args_hash, args, context)

# Wrap MCP tools with caching
def add_caching(tools):
    return [CachedMCPTool(tool) for tool in tools]

cached_mcp_tools = add_caching(mcp_tools)
```

### 3. Asynchronous Tool Execution

Execute multiple MCP tools concurrently:

```python
import asyncio
from jaf import Agent, run_streaming
from jaf.providers.mcp import create_mcp_stdio_tools

async def parallel_mcp_execution():
    tools = create_mcp_stdio_tools(
        command=["node", "async-server.js"]
    )
    
    agent = Agent(
        name="parallel_agent",
        instructions="Execute tools in parallel when possible",
        tools=tools
    )
    
    # Use streaming for better performance perception
    async for event in run_streaming(
        agent=agent,
        messages=[{"role": "user", "content": "Process multiple items"}]
    ):
        if event.type == "tool_call":
            print(f"Tool executing: {event.data.get('tool_name')}")
```

### 4. Server-side Batching

Batch multiple operations in a single MCP call:

```python
from pydantic import BaseModel
from typing import List

class BatchOperation(BaseModel):
    operations: List[Dict[str, Any]]
    batch_size: int = 10

class BatchMCPTool:
    def __init__(self, mcp_tool):
        self.mcp_tool = mcp_tool
        self.pending_operations = []
    
    async def execute_batch(self, operations: List[Dict[str, Any]]):
        # Send batch to MCP server
        batch_args = BatchOperation(operations=operations)
        return await self.mcp_tool.execute(batch_args, {})
    
    async def add_operation(self, operation: Dict[str, Any]):
        self.pending_operations.append(operation)
        
        # Execute when batch is full
        if len(self.pending_operations) >= 10:
            results = await self.execute_batch(self.pending_operations)
            self.pending_operations.clear()
            return results
```

## Performance Monitoring

### 1. Built-in Performance Tracking

```python
from jaf.core.performance import monitor_performance
from jaf import Agent, run

@monitor_performance("mcp_agent_execution")
def run_with_monitoring():
    agent = Agent(
        name="monitored_agent",
        instructions="Execute MCP tools with monitoring",
        tools=mcp_tools
    )
    
    return run(
        agent=agent,
        messages=[{"role": "user", "content": "Process with monitoring"}]
    )

# Get performance metrics
result = run_with_monitoring()
metrics = get_performance_summary()
print(f"Tool execution time: {metrics.tool_execution_time}")
print(f"IPC overhead: {metrics.ipc_overhead}")
```

### 2. Custom Performance Metrics

```python
import time
from contextlib import contextmanager

@contextmanager
def measure_mcp_performance():
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"MCP operation took {end_time - start_time:.3f}s")

# Usage
with measure_mcp_performance():
    result = run(agent=agent, messages=messages)
```

## Scaling MCP Integration

### 1. Load Balancing

Distribute MCP tool calls across multiple server instances:

```python
import random
from typing import List

class LoadBalancedMCPTools:
    def __init__(self, server_configs: List[Dict]):
        self.tool_pools = [
            create_mcp_stdio_tools(**config) 
            for config in server_configs
        ]
    
    def get_tools(self):
        # Round-robin or random selection
        return random.choice(self.tool_pools)

# Multiple MCP server instances
load_balanced_tools = LoadBalancedMCPTools([
    {"command": ["node", "server1.js"]},
    {"command": ["node", "server2.js"]},
    {"command": ["node", "server3.js"]},
])
```

### 2. Resource Limits

Prevent MCP tools from consuming excessive resources:

```python
import resource
import signal
from contextlib import contextmanager

@contextmanager
def resource_limited_execution(cpu_limit=30, memory_limit=512*1024*1024):
    """Limit CPU and memory for MCP tool execution."""
    old_cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
    old_memory_limit = resource.getrlimit(resource.RLIMIT_AS)
    
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, old_cpu_limit)
        resource.setrlimit(resource.RLIMIT_AS, old_memory_limit)

# Usage with resource limits
with resource_limited_execution():
    result = run(agent=agent, messages=messages)
```

## Performance Best Practices

### 1. Server Configuration

Optimize MCP server startup and execution:

```javascript
// MCP Server (Node.js example)
// Use cluster for CPU-intensive operations
const cluster = require('cluster');
const numCPUs = require('os').cpus().length;

if (cluster.isMaster) {
    // Fork workers equal to CPU count
    for (let i = 0; i < numCPUs; i++) {
        cluster.fork();
    }
} else {
    // Worker process handles MCP requests
    startMCPServer();
}
```

### 2. Tool Design

Design tools for optimal performance:

```python
# GOOD - Efficient tool design
class EfficientFileTool(BaseModel):
    path: str
    max_size: int = 1024 * 1024  # 1MB limit
    
    def execute(self, args, context):
        # Stream large files instead of loading into memory
        if os.path.getsize(args.path) > args.max_size:
            return {"error": "File too large", "size": os.path.getsize(args.path)}
        
        # Process in chunks for better memory usage
        with open(args.path, 'rb') as f:
            content = f.read()
        
        return {"content": content.decode('utf-8', errors='ignore')}

# BAD - Inefficient tool design
class InEfficientFileTool(BaseModel):
    def execute(self, args, context):
        # Loads entire file into memory
        return {"content": open(args.path).read()}
```

### 3. Error Recovery

Implement fast failure and recovery mechanisms:

```python
from jaf.core.errors import with_error_handling, CircuitBreaker

# Circuit breaker for failing MCP tools
mcp_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception
)

@with_error_handling("mcp_tool")
@mcp_circuit_breaker
async def resilient_mcp_call(tool, args, context):
    return await tool.execute(args, context)
```

## Troubleshooting Performance Issues

### Common Performance Problems

1. **Slow Tool Startup**
   - Solution: Use connection pooling
   - Check: Server startup time

2. **Memory Leaks in MCP Servers**
   - Solution: Regular server restarts
   - Monitor: Process memory usage

3. **Network Latency (SSE/HTTP)**
   - Solution: Use connection keep-alive
   - Consider: Server geographical proximity

4. **Blocking Tool Operations**
   - Solution: Implement timeouts
   - Use: Asynchronous execution

### Diagnostic Commands

```python
# Monitor MCP tool performance
def diagnose_mcp_performance(tools):
    for tool in tools:
        start_time = time.time()
        try:
            # Test with minimal args
            result = tool.execute({}, {})
            latency = time.time() - start_time
            print(f"Tool {tool.schema.name}: {latency:.3f}s")
        except Exception as e:
            print(f"Tool {tool.schema.name}: ERROR - {e}")
```

## Related Documentation

- [MCP Integration Overview](mcp.md)
- [MCP Examples](mcp-examples.md)
- [Performance Monitoring](performance-monitoring.md)
- [Error Handling](error-handling.md)