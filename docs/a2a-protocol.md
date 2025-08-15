# A2A Protocol - Agent-to-Agent Communication

JAF provides a complete implementation of the A2A (Agent-to-Agent) protocol, enabling distributed agent communication through JSON-RPC over HTTP. This protocol allows agents to communicate seamlessly across different services and applications.

## Overview

The A2A protocol is built on JSON-RPC 2.0 and provides:

- **Distributed Agents**: Agents running on different services can communicate directly
- **Task Management**: Submit, track, and cancel long-running tasks
- **Real-time Streaming**: Stream responses for iterative tasks
- **Agent Discovery**: Automatically discover available agents and their capabilities
- **Standard Protocol**: Based on JSON-RPC 2.0 for broad compatibility

## Protocol Specification

### Supported Methods

| Method | Description | Response Type |
|--------|-------------|---------------|
| `message/send` | Send a message to an agent | Immediate response |
| `message/stream` | Stream a message response | Server-sent events |
| `tasks/get` | Get task status and results | Task information |
| `tasks/cancel` | Cancel a running task | Cancellation status |
| `agent/getAuthenticatedExtendedCard` | Get agent capabilities | Agent card |

### Transport

- **Protocol**: JSON-RPC 2.0 over HTTP
- **Content-Type**: `application/json`
- **Streaming**: Server-sent events for `message/stream`

## Quick Start

### 1. Create A2A Client

```python
from jaf.a2a import A2A, connect_to_a2a_agent

# Simple client
client = A2A.client("http://localhost:3000")

# Full-featured connection
connection = await connect_to_a2a_agent("http://localhost:3000")
```

### 2. Send Messages

```python
import asyncio
from jaf.a2a import send_message_to_agent

async def demo():
    # Send a message to a specific agent
    response = await send_message_to_agent(
        client,
        agent_name="MathTutor", 
        message="What is 15 * 7?"
    )
    
    print(f"Response: {response}")

asyncio.run(demo())
```

### 3. Stream Responses

```python
from jaf.a2a import stream_message_to_agent

async def stream_demo():
    async for event in stream_message_to_agent(
        client,
        agent_name="ResearchAgent",
        message="Research the history of Python programming"
    ):
        if event.get("kind") == "message":
            print(f"Streamed: {event['message']['content']}")

asyncio.run(stream_demo())
```

## Agent Creation

### Basic Agent Setup

```python
from jaf.a2a import create_a2a_agent, create_a2a_tool

# Define a tool
def calculate(expression: str) -> str:
    """Safe calculation tool"""
    try:
        # Basic validation
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return 'Error: Invalid characters'
        return str(eval(expression))
    except:
        return 'Error: Invalid expression'

# Create A2A tool
calc_tool = create_a2a_tool(
    name="calculate",
    description="Perform mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    },
    execute_func=calculate
)

# Create A2A agent
math_agent = create_a2a_agent(
    name="MathTutor",
    description="A helpful math tutor agent",
    instruction="You are a math tutor. Use the calculate tool for math problems.",
    tools=[calc_tool]
)
```

### Transform to JAF Agent

```python
from jaf.a2a import transform_a2a_agent_to_jaf

# Convert A2A agent to JAF agent for local execution
jaf_agent = transform_a2a_agent_to_jaf(math_agent)

# Now you can use it with JAF's run engine
from jaf import run, RunConfig, RunState
```

## Server Setup

### Basic A2A Server

```python
import asyncio
from jaf.a2a import create_server_config, start_a2a_server

async def main():
    # Create multiple agents
    agents = {
        "MathTutor": math_agent,
        "ChatBot": chat_agent,
        "Assistant": assistant_agent
    }
    
    # Create server configuration
    server_config = create_server_config(
        agents=agents,
        name="Multi-Agent Server",
        description="A server with multiple specialized agents",
        port=3000,
        cors=True
    )
    
    # Start the server
    server = await start_a2a_server(server_config)
    print("A2A server running on http://localhost:3000")
    
    # Server provides these endpoints automatically:
    # GET  /.well-known/agent-card     # Agent discovery
    # POST /a2a                        # Main A2A endpoint
    # POST /a2a/agents/{agent_name}    # Agent-specific endpoint
    # GET  /a2a/health                 # Health check

asyncio.run(main())
```

### Advanced Server Configuration

```python
from jaf.a2a import create_a2a_server_config

config = create_a2a_server_config(
    agents=agents,
    server_info={
        "name": "Production Agent Server",
        "description": "Enterprise agent services",
        "version": "1.0.0",
        "contact": {"email": "admin@example.com"},
        "capabilities": {
            "streaming": True,
            "taskManagement": True,
            "authentication": True
        }
    },
    network_config={
        "host": "0.0.0.0",
        "port": 8080,
        "cors": {
            "allow_origins": ["https://app.example.com"],
            "allow_credentials": True
        }
    },
    memory_config={
        "provider": "redis",
        "url": "redis://localhost:6379",
        "task_ttl": 3600  # 1 hour
    }
)
```

## Agent Discovery

### Get Agent Capabilities

```python
from jaf.a2a import get_agent_card, discover_agents

async def discovery_demo():
    # Get specific agent information
    agent_card = await get_agent_card("http://localhost:3000")
    print(f"Available skills: {len(agent_card['skills'])}")
    
    # Discover all agents
    agents = await discover_agents("http://localhost:3000")
    for agent in agents:
        print(f"Agent: {agent['name']} - {agent['description']}")

asyncio.run(discovery_demo())
```

### Agent Card Structure

```json
{
  "name": "Multi-Agent Server",
  "description": "A server with multiple specialized agents",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "skills": [
    {
      "id": "math-calculation",
      "name": "Mathematical Calculations", 
      "description": "Perform arithmetic calculations and explain math concepts",
      "tags": ["math", "calculation", "education"],
      "examples": [
        {
          "query": "What is 15 * 7?",
          "result": "15 × 7 equals 105. This is a basic multiplication..."
        }
      ]
    }
  ],
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  },
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"]
}
```

## Task Management

### Submit and Track Tasks

```python
from jaf.a2a import create_a2a_task, create_message_request

async def task_demo():
    # Create a task request
    request = create_message_request(
        method="message/send",
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Generate a detailed report on Python performance"}],
            "messageId": "task_001",
            "contextId": "research_session",
            "kind": "message"
        }
    )
    
    # Send the request
    response = await send_a2a_request("http://localhost:3000", request)
    
    if "result" in response:
        task_id = response["result"]["taskId"]
        print(f"Task submitted: {task_id}")
        
        # Check task status
        status_request = {
            "jsonrpc": "2.0",
            "id": "status_check",
            "method": "tasks/get",
            "params": {"id": task_id}
        }
        
        status_response = await send_a2a_request(
            "http://localhost:3000", 
            status_request
        )
        
        task_info = status_response["result"]
        print(f"Status: {task_info['status']['state']}")

asyncio.run(task_demo())
```

## Streaming Communication

### Real-time Streaming

```python
from jaf.a2a import stream_message, parse_sse_event

async def streaming_demo():
    # Create streaming request
    stream_request = {
        "jsonrpc": "2.0",
        "id": "stream_001", 
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Write a story step by step"}],
                "messageId": "story_request",
                "contextId": "creative_session",
                "kind": "message"
            }
        }
    }
    
    # Stream the response
    async for raw_event in stream_message(
        "http://localhost:3000",
        stream_request
    ):
        event = parse_sse_event(raw_event)
        
        if event and event.get("kind") == "message":
            content = event["message"]["content"]
            print(f"Stream chunk: {content}")
        elif event and event.get("kind") == "status-update":
            print(f"Status: {event['status']['state']}")

asyncio.run(streaming_demo())
```

## Error Handling

### Robust Error Management

```python
from jaf.a2a import A2AError, A2AErrorCodes, send_message

async def error_handling_demo():
    try:
        response = await send_message(
            client,
            "Perform an impossible task"
        )
    except A2AError as e:
        if e.code == A2AErrorCodes.AGENT_NOT_FOUND:
            print(f"Agent not available: {e.message}")
        elif e.code == A2AErrorCodes.INVALID_REQUEST:
            print(f"Request error: {e.message}")
        elif e.code == A2AErrorCodes.EXECUTION_ERROR:
            print(f"Agent execution failed: {e.message}")
        else:
            print(f"A2A error: {e}")
    except Exception as e:
        print(f"Network or other error: {e}")

asyncio.run(error_handling_demo())
```

## Integration with JAF Core

### Hybrid Local/Remote Agents

```python
from jaf import Agent, run, RunConfig, RunState
from jaf.a2a import create_a2a_client, transform_a2a_agent_to_jaf

async def hybrid_demo():
    # Local JAF agent
    local_agent = Agent(
        name="LocalProcessor",
        instructions=lambda state: "Process data locally",
        tools=[]
    )
    
    # Remote A2A agent
    a2a_client = create_a2a_client("http://remote-server:3000")
    remote_agent = transform_a2a_agent_to_jaf(
        await a2a_client.get_agent("DataAnalyzer")
    )
    
    # Use both in JAF run configuration
    config = RunConfig(
        agent_registry={
            "LocalProcessor": local_agent,
            "RemoteAnalyzer": remote_agent
        },
        model_provider=make_litellm_provider("http://localhost:4000"),
        max_turns=5
    )
    
    # Agents can hand off to each other seamlessly
    initial_state = RunState(
        messages=[Message(role="user", content="Analyze this data")],
        current_agent_name="LocalProcessor",
        # ... other fields
    )
    
    result = await run(initial_state, config)

asyncio.run(hybrid_demo())
```

## Memory and Persistence

### Task Persistence

```python
from jaf.a2a.memory import create_a2a_in_memory_task_provider, A2AInMemoryTaskConfig

# Configure task persistence
memory_config = A2AInMemoryTaskConfig(
    max_tasks=1000,
    max_tasks_per_context=50,
    task_ttl_seconds=3600  # 1 hour
)

task_provider = create_a2a_in_memory_task_provider(memory_config)

# Tasks are automatically persisted and can be retrieved
# across server restarts (with Redis/PostgreSQL providers)
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["python", "-m", "jaf.a2a.examples.server_example"]
```

### Environment Configuration

```bash
# Server configuration
A2A_HOST=0.0.0.0
A2A_PORT=3000
A2A_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Memory configuration  
A2A_MEMORY_PROVIDER=redis
A2A_REDIS_URL=redis://redis:6379

# Task management
A2A_TASK_TTL=3600
A2A_MAX_TASKS_PER_CONTEXT=100

# Monitoring
A2A_ENABLE_METRICS=true
A2A_LOG_LEVEL=INFO
```

### Health Monitoring

```python
import httpx

async def health_check():
    """Monitor A2A server health"""
    try:
        response = await httpx.get("http://localhost:3000/a2a/health")
        health_data = response.json()
        
        if health_data.get("healthy"):
            print("✅ A2A server healthy")
            return True
        else:
            print(f"❌ A2A server unhealthy: {health_data}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
```

## Advanced Features

### Custom Protocol Handlers

```python
from jaf.a2a import create_protocol_handler_config

custom_config = create_protocol_handler_config(
    custom_methods={
        "custom/analyze": handle_custom_analyze,
        "custom/transform": handle_custom_transform
    },
    middleware=[
        authentication_middleware,
        rate_limiting_middleware,
        logging_middleware
    ]
)
```

### Multi-Agent Coordination

```python
async def coordination_demo():
    """Demonstrate multi-agent coordination"""
    
    # Agent 1: Data collector
    collector = create_a2a_agent(
        name="DataCollector",
        description="Collects and validates data",
        instruction="Collect data and pass to ProcessorAgent for analysis"
    )
    
    # Agent 2: Data processor  
    processor = create_a2a_agent(
        name="ProcessorAgent", 
        description="Processes and analyzes data",
        instruction="Process data and pass to ReporterAgent for final report"
    )
    
    # Agent 3: Report generator
    reporter = create_a2a_agent(
        name="ReporterAgent",
        description="Generates final reports",
        instruction="Generate comprehensive reports from processed data"
    )
    
    # Agents automatically coordinate through A2A protocol
    # Each agent can invoke the next in the pipeline
```

## Testing

### Unit Tests

```python
import pytest
from jaf.a2a import create_a2a_client, create_a2a_agent

@pytest.mark.asyncio
async def test_a2a_message_flow():
    """Test complete A2A message flow"""
    
    # Mock server setup
    mock_server = await create_mock_a2a_server()
    
    # Client creation
    client = create_a2a_client(mock_server.url)
    
    # Send test message
    response = await client.send_message("Hello, test agent!")
    
    # Verify response
    assert response["success"] is True
    assert "data" in response
    
    await mock_server.cleanup()
```

### Integration Tests

```python
@pytest.mark.integration
async def test_real_server_integration():
    """Test against real A2A server"""
    
    # Assumes test server running on localhost:3001
    client = create_a2a_client("http://localhost:3001")
    
    # Test agent discovery
    agents = await discover_agents(client.base_url)
    assert len(agents) > 0
    
    # Test message sending
    if "TestAgent" in [a["name"] for a in agents]:
        response = await send_message_to_agent(
            client, 
            "TestAgent", 
            "Integration test message"
        )
        assert response is not None
```

## Next Steps

- **[A2A Examples](a2a-examples.md)** - Comprehensive usage examples
- **[A2A API Reference](a2a-api-reference.md)** - Complete API documentation
- **[A2A Deployment Guide](a2a-deployment.md)** - Production deployment patterns
- **[A2A Protocol Specification](a2a-specification.md)** - Technical protocol details

---

The A2A protocol provides a robust foundation for distributed agent systems, enabling seamless communication between agents regardless of their hosting environment or implementation details.