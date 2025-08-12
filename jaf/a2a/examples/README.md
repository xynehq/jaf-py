# JAF A2A Examples

This directory contains comprehensive examples demonstrating the Agent-to-Agent (A2A) Communication Protocol implementation in JAF.

## Overview

The A2A protocol enables distributed agent communication through standardized JSON-RPC over HTTP, allowing agents to:
- Communicate across network boundaries
- Discover each other's capabilities
- Execute tasks collaboratively
- Stream real-time responses
- Handle complex multi-agent workflows

## Examples

### 1. Server Example (`server_example.py`)

A complete A2A server implementation showcasing:

- **Multiple specialized agents**:
  - `MathTutor` - Mathematical calculations with calculator tool
  - `WeatherBot` - Weather information with location lookup
  - `Translator` - Text translation between languages  
  - `Assistant` - General purpose conversational agent

- **Full A2A protocol support**:
  - Agent Card discovery (`.well-known/agent-card`)
  - JSON-RPC message handling (`/a2a`)
  - Agent-specific endpoints (`/a2a/agents/{name}`)
  - Health checks and capabilities
  - OpenAPI documentation

- **Production features**:
  - CORS support for web clients
  - Error handling and validation
  - Security checks for tool execution
  - Structured tool schemas with Pydantic

**Usage:**
```bash
python server_example.py
```

Server will start on `http://localhost:3000` with full API documentation at `/docs`.

### 2. Client Example (`client_example.py`)

Comprehensive client demonstrating:

- **Agent discovery**: Finding available agents and their capabilities
- **Direct communication**: Sending messages to specific agents
- **Streaming responses**: Real-time message streaming
- **Health monitoring**: Server health and capability checks
- **Error handling**: Robust error handling and recovery
- **Convenience methods**: High-level client abstractions

**Usage:**
```bash
# Start server first
python server_example.py

# Then run client
python client_example.py
```

### 3. Integration Example (`integration_example.py`)

End-to-end integration test showing:

- **Self-contained testing**: Server and client in one process
- **Complete workflow**: Discovery → Health → Communication → Cleanup
- **Multiple interaction patterns**: Different agents and message types
- **Result validation**: Comprehensive testing and reporting
- **Resource management**: Proper startup and shutdown

**Usage:**
```bash
python integration_example.py
```

## Quick Start

### Basic Server

```python
from jaf.a2a import A2A, create_a2a_agent, create_server_config, start_a2a_server

# Create agent
agent = A2A.agent("MyBot", "Helpful assistant", "You are helpful", [])

# Create and start server
config = A2A.server({"MyBot": agent}, "My Server", "Test server", 3000)
server = await A2A.start_server(config)
```

### Basic Client

```python
from jaf.a2a import A2A

# Connect to agent
connection = await A2A.connect("http://localhost:3000")

# Send message
response = await connection["ask"]("Hello, how can you help?")
print(response)

# Check health
health = await connection["health"]()
print(f"Status: {health['status']}")
```

## API Reference

### Core A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card` | GET | Agent discovery and capabilities |
| `/a2a` | POST | Main JSON-RPC endpoint |
| `/a2a/agents/{name}` | POST | Agent-specific messaging |
| `/a2a/health` | GET | Server health check |
| `/a2a/capabilities` | GET | Protocol capabilities |
| `/docs` | GET | OpenAPI documentation |

### JSON-RPC Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `message/send` | Send message to agent | `message`, `configuration` |
| `message/stream` | Stream message response | `message`, `configuration` |
| `tasks/get` | Retrieve task status | `id`, `historyLength` |
| `tasks/cancel` | Cancel running task | `id` |
| `agent/getAuthenticatedExtendedCard` | Get extended agent info | - |

### Message Format

```json
{
  "jsonrpc": "2.0",
  "id": "unique-request-id",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Hello!"}],
      "messageId": "msg_123",
      "contextId": "session_456",
      "kind": "message"
    },
    "configuration": {
      "model": "gpt-4",
      "temperature": 0.7
    }
  }
}
```

## Testing Examples

### Using curl

```bash
# Discover agents
curl http://localhost:3000/.well-known/agent-card

# Send message to math agent
curl -X POST http://localhost:3000/a2a/agents/MathTutor \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is 25 * 17?"}],
        "messageId": "msg_1",
        "contextId": "test_session",
        "kind": "message"
      }
    }
  }'

# Check server health
curl http://localhost:3000/a2a/health
```

### Using Python requests

```python
import requests

# Discover agents
response = requests.get("http://localhost:3000/.well-known/agent-card")
agent_card = response.json()
print(f"Found: {agent_card['name']}")

# Send message
message_request = {
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello!"}],
            "messageId": "msg_1",
            "contextId": "session_1",
            "kind": "message"
        }
    }
}

response = requests.post(
    "http://localhost:3000/a2a/agents/Assistant",
    json=message_request
)
result = response.json()
print(f"Response: {result}")
```

## Advanced Features

### Custom Tools

```python
from pydantic import BaseModel, Field
from jaf.a2a import create_a2a_tool

class MyToolArgs(BaseModel):
    input_text: str = Field(description="Text to process")

async def my_tool(args: MyToolArgs, context) -> dict:
    return {"result": f"Processed: {args.input_text}"}

tool = create_a2a_tool(
    "my_tool",
    "Custom processing tool", 
    MyToolArgs.model_json_schema(),
    my_tool
)
```

### Streaming Responses

```python
from jaf.a2a import stream_message, create_a2a_client

client = create_a2a_client("http://localhost:3000")

async for event in stream_message(client, "Tell me a story"):
    print(f"Event: {event}")
```

### Error Handling

```python
from jaf.a2a import send_message_to_agent, create_a2a_client

client = create_a2a_client("http://localhost:3000")

try:
    response = await send_message_to_agent(client, "UnknownAgent", "Hello")
except Exception as error:
    print(f"Error: {error}")
```

## Configuration

### Server Configuration

```python
server_config = {
    "agents": {/* agent dictionary */},
    "agentCard": {
        "name": "My Server",
        "description": "Server description",
        "version": "1.0.0",
        "provider": {
            "organization": "My Org",
            "url": "https://example.com"
        }
    },
    "port": 3000,
    "host": "localhost"
}
```

### Client Configuration

```python
client_config = {
    "timeout": 30000,  # 30 seconds
    "retries": 3,
    "headers": {
        "Authorization": "Bearer token"
    }
}

client = create_a2a_client("http://localhost:3000", client_config)
```

## Development

### Adding New Agents

1. Define tool schemas with Pydantic
2. Implement async tool functions
3. Create A2A tools with `create_a2a_tool`
4. Create agent with `create_a2a_agent`
5. Add to server configuration

### Testing

Run the integration example to verify your setup:

```bash
python integration_example.py
```

Expected output:
- ✅ Server startup
- ✅ Agent discovery
- ✅ Health checks
- ✅ Message interactions
- ✅ Cleanup

## Troubleshooting

### Common Issues

**Server won't start:**
- Check port availability: `lsof -i :3000`
- Verify dependencies: `pip install fastapi uvicorn`

**Client can't connect:**
- Ensure server is running
- Check firewall settings
- Verify URL format

**Agent not responding:**
- Check agent registration in server config
- Verify tool implementations
- Check server logs for errors

**Tool execution fails:**
- Validate tool argument schemas
- Check async/await usage
- Verify context parameter handling

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance

For production deployments:
- Use proper model providers (not mock)
- Implement persistent task storage
- Add authentication and rate limiting
- Scale with load balancers

## Contributing

When adding examples:
1. Follow the existing patterns
2. Include comprehensive error handling
3. Add proper documentation
4. Test with integration example
5. Update this README

## License

These examples are part of the JAF Framework and follow the same license terms.