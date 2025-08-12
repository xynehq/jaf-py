# Server API Reference

JAF provides a production-ready FastAPI server that exposes your agents via HTTP endpoints. This comprehensive reference covers all available endpoints, request/response formats, and usage examples.

## Quick Start

```python
from jaf import run_server, Agent, make_litellm_provider
from jaf.server.types import ServerConfig
from jaf.core.types import RunConfig

# Create your agents
agent = Agent(name="MyAgent", instructions=lambda state: "You are helpful.", tools=[])

# Configure the server
server_config = ServerConfig(
    host="127.0.0.1",
    port=3000,
    agent_registry={"MyAgent": agent},
    run_config=RunConfig(
        agent_registry={"MyAgent": agent},
        model_provider=make_litellm_provider("http://localhost:4000"),
        max_turns=5
    )
)

# Start the server
await run_server(server_config)
```

## Base URL and Authentication

- **Base URL**: `http://localhost:3000` (configurable)
- **Authentication**: None (implement via middleware if needed)
- **Content-Type**: `application/json` for all POST requests

## Core Endpoints

### Health Check

Check server health and get basic information.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.123456Z",
  "version": "2.0.0",
  "uptime": 45000
}
```

**Example**:
```bash
curl http://localhost:3000/health
```

**Response Fields**:
- `status`: Server health status (`"healthy"` or `"unhealthy"`)
- `timestamp`: Current server timestamp in ISO format
- `version`: JAF server version
- `uptime`: Server uptime in milliseconds

### List Agents

Get information about all available agents.

**Endpoint**: `GET /agents`

**Response**:
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "name": "MathTutor",
        "description": "You are a helpful math tutor. Use the calculator tool to perform calculations and explain math concepts clearly.",
        "tools": ["calculate"]
      },
      {
        "name": "ChatBot", 
        "description": "You are a friendly chatbot. Use the greeting tool when meeting new people, and engage in helpful conversation.",
        "tools": ["greet"]
      }
    ]
  }
}
```

**Example**:
```bash
curl http://localhost:3000/agents
```

**Response Fields**:
- `success`: Boolean indicating if request succeeded
- `data.agents`: Array of agent information objects
  - `name`: Agent identifier
  - `description`: Agent's instruction summary (truncated to 200 chars)
  - `tools`: List of available tool names

## Chat Endpoints

### Main Chat Endpoint

Send messages to any agent for processing.

**Endpoint**: `POST /chat`

**Request Body**:
```json
{
  "agent_name": "MathTutor",
  "messages": [
    {
      "role": "user",
      "content": "What is 15 * 7?"
    }
  ],
  "context": {
    "userId": "user-123",
    "permissions": ["user"]
  },
  "max_turns": 5,
  "conversation_id": "math-session-1",
  "stream": false
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "run_id": "run_12345",
    "trace_id": "trace_67890",
    "messages": [
      {
        "role": "user",
        "content": "What is 15 * 7?"
      },
      {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_123",
            "type": "function",
            "function": {
              "name": "calculate",
              "arguments": "{\"expression\": \"15 * 7\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "content": "15 * 7 = 105",
        "tool_call_id": "call_123"
      },
      {
        "role": "assistant",
        "content": "15 × 7 equals 105. This is a basic multiplication problem where we multiply 15 by 7 to get the result."
      }
    ],
    "outcome": {
      "status": "completed",
      "output": "15 × 7 equals 105. This is a basic multiplication problem where we multiply 15 by 7 to get the result."
    },
    "turn_count": 2,
    "execution_time_ms": 1250,
    "conversation_id": "math-session-1"
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "MathTutor",
    "messages": [{"role": "user", "content": "What is 15 * 7?"}],
    "context": {"userId": "demo", "permissions": ["user"]}
  }'
```

### Agent-Specific Chat Endpoint

Alternative endpoint that specifies the agent in the URL path.

**Endpoint**: `POST /agents/{agent_name}/chat`

**Request Body** (same as `/chat` but without `agent_name`):
```json
{
  "messages": [
    {
      "role": "user", 
      "content": "Hi, my name is Alice"
    }
  ],
  "context": {
    "userId": "user-456",
    "permissions": ["user"]
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:3000/agents/ChatBot/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, my name is Alice"}],
    "context": {"userId": "demo", "permissions": ["user"]}
  }'
```

## Request Parameters

### ChatRequest Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `agent_name` | string | Yes* | - | Agent to use for processing (* not required for agent-specific endpoint) |
| `messages` | array | Yes | - | Conversation messages |
| `context` | object | No | `{}` | Context data passed to agent and tools |
| `max_turns` | integer | No | `10` | Maximum conversation turns |
| `stream` | boolean | No | `false` | Enable streaming responses (not yet implemented) |
| `conversation_id` | string | No | auto-generated | ID for memory persistence |
| `memory` | object | No | `null` | Memory configuration override |

### Message Format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | Message role: `"user"`, `"assistant"`, `"system"`, or `"tool"` |
| `content` | string | Yes | Message content |
| `tool_call_id` | string | No | ID linking tool responses to tool calls |
| `tool_calls` | array | No | Tool calls made by assistant (auto-populated) |

### Tool Call Format

```json
{
  "id": "call_abc123",
  "type": "function", 
  "function": {
    "name": "calculate",
    "arguments": "{\"expression\": \"2 + 2\"}"
  }
}
```

## Memory Endpoints

### Get Conversation

Retrieve complete conversation history (requires memory provider).

**Endpoint**: `GET /conversations/{conversation_id}`

**Response**:
```json
{
  "success": true,
  "data": {
    "conversation_id": "user-123-session-1",
    "user_id": "user-123",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      },
      {
        "role": "assistant", 
        "content": "Hi there! How can I help you today?"
      }
    ],
    "metadata": {
      "session_start": "2024-01-15T10:00:00Z",
      "topic": "general_chat"
    }
  }
}
```

**Example**:
```bash
curl http://localhost:3000/conversations/user-123-session-1
```

**Error Response** (conversation not found):
```json
{
  "success": false,
  "error": "Conversation user-123-session-1 not found"
}
```

### Delete Conversation

Delete a conversation from memory.

**Endpoint**: `DELETE /conversations/{conversation_id}`

**Response**:
```json
{
  "success": true,
  "data": {
    "conversation_id": "user-123-session-1",
    "deleted": true
  }
}
```

**Example**:
```bash
curl -X DELETE http://localhost:3000/conversations/user-123-session-1
```

### Memory Health Check

Check memory provider health and performance.

**Endpoint**: `GET /memory/health`

**Response**:
```json
{
  "success": true,
  "data": {
    "healthy": true,
    "provider": "RedisMemoryProvider",
    "latency_ms": 2.5,
    "details": {
      "connections": 5,
      "memory_usage": "15.2MB",
      "version": "7.0.0"
    }
  }
}
```

**Example**:
```bash
curl http://localhost:3000/memory/health
```

## Response Format

All endpoints follow a consistent response format:

### Success Response
```json
{
  "success": true,
  "data": {
    // Endpoint-specific data
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": "Detailed error message"
}
```

## Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid request format or parameters |
| 404 | Not Found | Agent or conversation not found |
| 500 | Internal Server Error | Server or agent execution error |

## Advanced Usage Examples

### Persistent Conversation

Start and continue a conversation with memory:

```bash
# Start conversation
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "ChatBot",
    "messages": [{"role": "user", "content": "Hello, I am starting a new conversation"}],
    "agent_name": "ChatBot",
    "conversation_id": "my-conversation",
    "context": {"userId": "demo", "permissions": ["user"]}
  }'

# Continue conversation
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "ChatBot", 
    "messages": [{"role": "user", "content": "Do you remember me?"}],
    "conversation_id": "my-conversation",
    "context": {"userId": "demo", "permissions": ["user"]}
  }'

# Get conversation history  
curl http://localhost:3000/conversations/my-conversation

# Delete conversation
curl -X DELETE http://localhost:3000/conversations/my-conversation
```

### Multi-Tool Agent Interaction

Use an agent with multiple tools:

```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "Assistant",
    "messages": [{"role": "user", "content": "Calculate 25 + 17 and then greet me as Bob"}],
    "context": {"userId": "demo", "permissions": ["user"]}
  }'
```

### Complex Context Usage

Pass rich context data to agents:

```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "CustomerService",
    "messages": [{"role": "user", "content": "I need help with my account"}],
    "context": {
      "userId": "user-12345",
      "accountId": "acc-67890", 
      "permissions": ["user", "account_access"],
      "location": "US",
      "language": "en",
      "tier": "premium"
    }
  }'
```

## Error Handling

### Common Error Scenarios

**Agent Not Found**:
```json
{
  "success": false,
  "error": "Agent 'NonExistentAgent' not found. Available agents: MathTutor, ChatBot, Assistant"
}
```

**Invalid Message Format**:
```json
{
  "success": false,
  "error": "1 validation error for ChatRequest\nmessages.0.role\n  Input should be 'user', 'assistant', 'system' or 'tool'"
}
```

**Memory Not Configured**:
```json
{
  "success": false,
  "error": "Memory not configured for this server"
}
```

**Tool Execution Error**:
```json
{
  "success": true,
  "data": {
    "outcome": {
      "status": "error",
      "error": {
        "type": "ToolExecutionError",
        "message": "Calculator tool failed: Invalid expression"
      }
    }
  }
}
```

## Server Configuration

### Basic Configuration

```python
from jaf.server.types import ServerConfig

config = ServerConfig(
    host="127.0.0.1",           # Bind address
    port=3000,                  # Port number
    agent_registry=agents,      # Agent dictionary
    run_config=run_config,      # JAF run configuration
    cors=True                   # Enable CORS (all origins)
)
```

### CORS Configuration

```python
# Disable CORS
config = ServerConfig(cors=False, ...)

# Custom CORS settings
config = ServerConfig(
    cors={
        "allow_origins": ["https://myapp.com", "https://admin.myapp.com"],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    },
    ...
)
```

### Production Configuration

```python
config = ServerConfig(
    host="0.0.0.0",             # Listen on all interfaces
    port=int(os.getenv("PORT", "8000")),
    agent_registry=agents,
    run_config=RunConfig(
        agent_registry=agents,
        model_provider=provider,
        max_turns=10,
        memory=memory_config,      # Enable persistence
        on_event=trace_collector.collect  # Enable tracing
    ),
    cors={
        "allow_origins": [os.getenv("FRONTEND_URL")],
        "allow_credentials": True
    }
)
```

## Monitoring and Observability

### Request Logging

The server automatically logs all requests:

```
[JAF:SERVER] POST /chat - 200 - 1.250s
[JAF:SERVER] GET /agents - 200 - 0.045s
[JAF:SERVER] GET /health - 200 - 0.012s
```

### Metrics Endpoint

Basic metrics are available at `/metrics`:

```bash
curl http://localhost:3000/metrics
```

**Response**:
```json
{
  "status": "ok"
}
```

### Custom Middleware

Add custom monitoring middleware:

```python
from fastapi import Request
import time

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log to your monitoring system
    logger.info(f"Request processed", extra={
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration": process_time
    })
    
    return response
```

## API Documentation

The server provides interactive API documentation:

- **Swagger UI**: `http://localhost:3000/docs`
- **ReDoc**: `http://localhost:3000/redoc`

These interfaces allow you to:
- Browse all available endpoints
- View request/response schemas
- Test endpoints directly in the browser
- Download OpenAPI specifications

## Client Libraries

### Python Client Example

```python
import httpx
import asyncio

class JAFClient:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def chat(self, agent_name: str, message: str, context: dict = None, conversation_id: str = None):
        """Send a message to an agent."""
        payload = {
            "agent_name": agent_name,
            "messages": [{"role": "user", "content": message}],
            "context": context or {},
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        response = await self.client.post(f"{self.base_url}/chat", json=payload)
        return response.json()
    
    async def list_agents(self):
        """Get list of available agents."""
        response = await self.client.get(f"{self.base_url}/agents")
        return response.json()
    
    async def get_conversation(self, conversation_id: str):
        """Get conversation history."""
        response = await self.client.get(f"{self.base_url}/conversations/{conversation_id}")
        return response.json()

# Usage
client = JAFClient()
result = await client.chat("MathTutor", "What is 2 + 2?")
print(result)
```

### JavaScript/Node.js Client Example

```javascript
class JAFClient {
    constructor(baseUrl = 'http://localhost:3000') {
        this.baseUrl = baseUrl;
    }
    
    async chat(agentName, message, context = {}, conversationId = null) {
        const payload = {
            agent_name: agentName,
            messages: [{ role: 'user', content: message }],
            context: context
        };
        
        if (conversationId) {
            payload.conversation_id = conversationId;
        }
        
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        return response.json();
    }
    
    async listAgents() {
        const response = await fetch(`${this.baseUrl}/agents`);
        return response.json();
    }
}

// Usage
const client = new JAFClient();
const result = await client.chat('MathTutor', 'What is 2 + 2?');
console.log(result);
```

## Performance Considerations

### Request Timeout

Configure appropriate timeouts for your use case:

```python
# Client-side timeout
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.post(url, json=data)
```

### Connection Pooling

For high-throughput applications:

```python
# Reuse client connections
client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20
    )
)
```

### Batch Processing

Process multiple requests efficiently:

```python
async def process_batch(messages):
    tasks = []
    for msg in messages:
        task = client.chat("Agent", msg)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## Security Considerations

### Input Validation

The server validates all input using Pydantic models, but consider additional validation:

```python
def validate_context(context: dict) -> dict:
    """Additional context validation."""
    # Remove sensitive fields
    safe_context = {k: v for k, v in context.items() if not k.startswith('_')}
    
    # Validate user permissions
    if 'permissions' in safe_context:
        allowed_permissions = {'user', 'admin', 'read', 'write'}
        safe_context['permissions'] = [
            p for p in safe_context['permissions'] 
            if p in allowed_permissions
        ]
    
    return safe_context
```

### Rate Limiting

Implement rate limiting for production:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    # ... endpoint implementation
```

### Authentication

Add authentication middleware:

```python
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Skip auth for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Check API key
    api_key = request.headers.get("Authorization")
    if not api_key or not validate_api_key(api_key):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing API key"}
        )
    
    return await call_next(request)
```

## Next Steps

- Explore [Examples](examples.md) for real-world server implementations
- Learn about [Deployment](deployment.md) for production setup
- Check [Memory System](memory-system.md) for persistence configuration
- Review [Troubleshooting](troubleshooting.md) for common server issues