# A2A API Reference

Complete API documentation for the JAF Agent-to-Agent (A2A) Communication Protocol.

## Overview

The A2A API provides JSON-RPC 2.0 based communication between agents over HTTP. This reference covers all available endpoints, request/response formats, and error handling.

## Base Endpoints

### Health Check

**GET** `/.well-known/agent-card`

Returns the agent card describing capabilities and available agents.

```json
{
  "name": "JAF A2A Server",
  "description": "Multi-agent server supporting A2A protocol",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "skills": [
    {
      "id": "math_tutor",
      "name": "Math Tutor",
      "description": "Mathematical calculations and explanations",
      "tags": ["math", "calculation", "education"]
    }
  ],
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true
  }
}
```

**GET** `/a2a/health`

Basic health check endpoint.

```json
{
  "status": "healthy",
  "timestamp": "2024-03-15T14:30:00Z",
  "version": "1.0.0"
}
```

## JSON-RPC Endpoints

All A2A communication uses JSON-RPC 2.0 over HTTP POST.

### Base URL: `/a2a`

### Supported Methods

#### message/send

Send a message to the default agent or routing system.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "msg_123",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Hello, can you help me with math?"
        }
      ],
      "messageId": "user_msg_001",
      "contextId": "session_123",
      "kind": "message"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "msg_123",
  "result": {
    "kind": "completion",
    "taskId": "task_456",
    "contextId": "session_123",
    "message": {
      "role": "agent",
      "parts": [
        {
          "kind": "text",
          "text": "I'd be happy to help you with math! What specific problem would you like to work on?"
        }
      ],
      "messageId": "agent_msg_001",
      "contextId": "session_123",
      "kind": "message"
    },
    "final": true
  }
}
```

#### message/stream

Stream a message response for real-time interaction.

**Request:** Same as `message/send`

**Response:** Server-Sent Events (SSE) stream with `Content-Type: text/event-stream`

```
data: {"jsonrpc": "2.0", "id": "msg_123", "result": {"kind": "status-update", "taskId": "task_456", "status": {"state": "working", "timestamp": "2024-03-15T14:30:01Z"}, "final": false}}

data: {"jsonrpc": "2.0", "id": "msg_123", "result": {"kind": "completion", "taskId": "task_456", "message": {"role": "agent", "parts": [{"kind": "text", "text": "I can help with that calculation..."}], "messageId": "agent_msg_001", "contextId": "session_123", "kind": "message"}, "final": true}}
```

#### tasks/get

Retrieve task status and results.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "task_query_123",
  "method": "tasks/get",
  "params": {
    "id": "task_456"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "task_query_123",
  "result": {
    "id": "task_456",
    "contextId": "session_123",
    "kind": "task",
    "status": {
      "state": "completed",
      "message": {
        "role": "agent",
        "parts": [
          {
            "kind": "text",
            "text": "The calculation result is 42."
          }
        ],
        "messageId": "agent_msg_001",
        "contextId": "session_123",
        "kind": "message"
      },
      "timestamp": "2024-03-15T14:30:05Z"
    }
  }
}
```

#### tasks/cancel

Cancel a running task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "cancel_123",
  "method": "tasks/cancel",
  "params": {
    "id": "task_456"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "cancel_123",
  "result": {
    "id": "task_456",
    "cancelled": true,
    "timestamp": "2024-03-15T14:30:02Z"
  }
}
```

### Agent-Specific Endpoints

#### Base URL: `/a2a/agents/{agentName}`

#### GET `/a2a/agents/{agentName}/card`

Get agent-specific capabilities.

```json
{
  "name": "MathTutor",
  "description": "Specialized mathematical assistant",
  "version": "1.0.0",
  "skills": [
    {
      "id": "calculate",
      "name": "Calculate",
      "description": "Perform mathematical calculations",
      "tags": ["math", "calculation"]
    }
  ]
}
```

#### POST `/a2a/agents/{agentName}`

Send message directly to specific agent. Supports same methods as base endpoint:
- `message/send`
- `message/stream`
- `tasks/get`
- `tasks/cancel`

## Data Types

### A2AMessage

```typescript
interface A2AMessage {
  role: "user" | "agent" | "system";
  parts: A2APart[];
  messageId: string;
  contextId: string;
  kind: "message";
  timestamp?: string;
}
```

### A2APart

```typescript
type A2APart = A2ATextPart | A2ADataPart;

interface A2ATextPart {
  kind: "text";
  text: string;
}

interface A2ADataPart {
  kind: "data";
  data: any;
  mimeType?: string;
}
```

### A2ATask

```typescript
interface A2ATask {
  id: string;
  contextId: string;
  kind: "task";
  status: A2ATaskStatus;
}
```

### A2ATaskStatus

```typescript
interface A2ATaskStatus {
  state: "submitted" | "working" | "completed" | "failed" | "cancelled";
  message?: A2AMessage;
  timestamp: string;
  error?: A2AError;
}
```

### A2AError

```typescript
interface A2AError {
  code: number;
  message: string;
  data?: any;
}
```

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON was received |
| -32600 | Invalid Request | JSON-RPC request was invalid |
| -32601 | Method Not Found | Method does not exist |
| -32602 | Invalid Params | Invalid method parameters |
| -32603 | Internal Error | Internal JSON-RPC error |
| -32000 | Agent Not Found | Specified agent does not exist |
| -32001 | Task Not Found | Specified task does not exist |
| -32002 | Agent Unavailable | Agent is temporarily unavailable |
| -32003 | Rate Limited | Too many requests |
| -32004 | Authentication Required | Request requires authentication |
| -32005 | Permission Denied | Insufficient permissions |

## Error Response Format

```json
{
  "jsonrpc": "2.0",
  "id": "msg_123",
  "error": {
    "code": -32000,
    "message": "Agent not found",
    "data": {
      "agentName": "NonExistentAgent",
      "availableAgents": ["MathTutor", "ChatBot", "Assistant"]
    }
  }
}
```

## Authentication

Currently, the A2A protocol supports basic authentication through headers:

```http
Authorization: Bearer <token>
X-API-Key: <api-key>
```

## Rate Limiting

Responses include rate limit headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## CORS Support

The server supports CORS for browser-based clients:

```http
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key
```

## Client Libraries

### Python

```python
from jaf.a2a import create_a2a_client, send_message

client = create_a2a_client("http://localhost:3000")
response = await send_message(client, "Hello, world!")
```

### JavaScript

```javascript
const response = await fetch('/a2a', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: '1',
    method: 'message/send',
    params: { message: { /* A2AMessage */ } }
  })
});
```

### cURL

```bash
curl -X POST http://localhost:3000/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Hello"}],
        "messageId": "msg_1",
        "contextId": "session_1",
        "kind": "message"
      }
    }
  }'
```

## Best Practices

1. **Use meaningful IDs**: Always provide unique, meaningful request IDs
2. **Handle streaming properly**: For `message/stream`, properly parse SSE events
3. **Implement retry logic**: Handle temporary failures with exponential backoff
4. **Validate responses**: Always check for error responses
5. **Context management**: Use consistent `contextId` for conversation continuity
6. **Resource cleanup**: Cancel tasks when no longer needed

## Related Documentation

- [A2A Protocol Overview](a2a-protocol.md)
- [A2A Examples](a2a-examples.md)
- [A2A Deployment Guide](a2a-deployment.md)