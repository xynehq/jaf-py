# A2A Protocol Specification

Technical specification for the JAF Agent-to-Agent (A2A) Communication Protocol v0.3.0.

## Overview

The A2A protocol enables structured communication between AI agents using JSON-RPC 2.0 over HTTP. This specification defines the protocol structure, message formats, state management, and interaction patterns.

## Protocol Information

- **Version**: 0.3.0
- **Transport**: HTTP/HTTPS
- **Message Format**: JSON-RPC 2.0
- **Content Type**: `application/json`
- **Character Encoding**: UTF-8

## Base Protocol Stack

```
┌─────────────────────────────────────┐
│           Application Layer         │
│        (Agent Implementations)     │
├─────────────────────────────────────┤
│           A2A Protocol Layer        │
│     (Message Format & Routing)     │
├─────────────────────────────────────┤
│          JSON-RPC 2.0 Layer        │
│       (Request/Response Format)    │
├─────────────────────────────────────┤
│           HTTP/HTTPS Layer          │
│      (Transport & Security)        │
└─────────────────────────────────────┘
```

## Core Message Types

### A2AMessage

The fundamental communication unit between agents.

```json
{
  "role": "user" | "agent" | "system",
  "parts": [
    {
      "kind": "text",
      "text": "Hello, how can you help me?"
    }
  ],
  "messageId": "unique-message-identifier",
  "contextId": "conversation-context-identifier",
  "kind": "message",
  "timestamp": "2024-03-15T14:30:00Z"
}
```

**Fields:**
- `role`: Message sender role (required)
- `parts`: Array of message parts (required, min 1 item)
- `messageId`: Unique message identifier (required)
- `contextId`: Conversation context identifier (required)
- `kind`: Always "message" (required)
- `timestamp`: ISO 8601 timestamp (optional)

### A2APart Types

#### Text Part
```json
{
  "kind": "text",
  "text": "The message content as a string"
}
```

#### Data Part
```json
{
  "kind": "data",
  "data": { "any": "structured data" },
  "mimeType": "application/json"
}
```

### A2ATask

Represents an ongoing operation or conversation state.

```json
{
  "id": "task-unique-identifier",
  "contextId": "conversation-context-identifier", 
  "kind": "task",
  "status": {
    "state": "submitted" | "working" | "completed" | "failed" | "cancelled",
    "message": { /* A2AMessage */ },
    "timestamp": "2024-03-15T14:30:00Z",
    "error": { /* A2AError (optional) */ }
  }
}
```

**Task States:**
- `submitted`: Task received and queued
- `working`: Task being processed
- `completed`: Task finished successfully
- `failed`: Task failed with error
- `cancelled`: Task cancelled by request

### A2AError

Structured error information.

```json
{
  "code": -32000,
  "message": "Human-readable error description",
  "data": {
    "additional": "error context",
    "errorType": "AgentNotFound",
    "timestamp": "2024-03-15T14:30:00Z"
  }
}
```

## JSON-RPC 2.0 Method Specifications

### message/send

Send a message for processing.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "request-identifier",
  "method": "message/send",
  "params": {
    "message": { /* A2AMessage */ },
    "configuration": {
      "maxTurns": 10,
      "timeout": 30000,
      "priority": "normal"
    }
  }
}
```

**Success Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "request-identifier",
  "result": {
    "kind": "completion",
    "taskId": "generated-task-id",
    "contextId": "conversation-context",
    "message": { /* A2AMessage response */ },
    "final": true
  }
}
```

**Error Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "request-identifier",
  "error": {
    "code": -32000,
    "message": "Agent not found",
    "data": {
      "agentName": "RequestedAgent",
      "availableAgents": ["Agent1", "Agent2"]
    }
  }
}
```

### message/stream

Stream a message response with real-time updates.

**Request:** Same as `message/send`

**Response:** Server-Sent Events stream

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"jsonrpc":"2.0","id":"req-1","result":{"kind":"status-update","taskId":"task-123","status":{"state":"working","timestamp":"2024-03-15T14:30:01Z"},"final":false}}

data: {"jsonrpc":"2.0","id":"req-1","result":{"kind":"partial-completion","taskId":"task-123","message":{"role":"agent","parts":[{"kind":"text","text":"I'm thinking about"}],"messageId":"resp-1","contextId":"ctx-1","kind":"message"},"final":false}}

data: {"jsonrpc":"2.0","id":"req-1","result":{"kind":"completion","taskId":"task-123","message":{"role":"agent","parts":[{"kind":"text","text":"I'm thinking about your question. Here's my response..."}],"messageId":"resp-1","contextId":"ctx-1","kind":"message"},"final":true}}
```

### tasks/get

Retrieve task information and status.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "query-identifier",
  "method": "tasks/get",
  "params": {
    "id": "task-identifier"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "query-identifier",
  "result": {
    "id": "task-identifier",
    "contextId": "conversation-context",
    "kind": "task",
    "status": {
      "state": "completed",
      "message": { /* Final A2AMessage */ },
      "timestamp": "2024-03-15T14:30:05Z"
    }
  }
}
```

### tasks/cancel

Cancel an active task.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "cancel-identifier",
  "method": "tasks/cancel",
  "params": {
    "id": "task-identifier",
    "reason": "User requested cancellation"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "cancel-identifier",
  "result": {
    "id": "task-identifier",
    "cancelled": true,
    "timestamp": "2024-03-15T14:30:02Z"
  }
}
```

## Agent Discovery

### Agent Card Format

Agents expose capabilities through standardized agent cards.

```json
{
  "name": "MathTutor",
  "description": "Specialized mathematical assistant",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "url": "https://api.example.com/agents/mathtutor",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true,
    "contextAccumulation": true
  },
  "skills": [
    {
      "id": "calculate",
      "name": "Mathematical Calculation",
      "description": "Perform arithmetic and algebraic calculations",
      "tags": ["math", "calculation", "arithmetic"],
      "examples": [
        {
          "input": "What is 15 * 7?",
          "output": "15 * 7 = 105"
        }
      ]
    }
  ],
  "inputModes": ["text", "data"],
  "outputModes": ["text", "data"],
  "authentication": {
    "required": false,
    "schemes": ["bearer", "apikey"]
  },
  "rateLimit": {
    "requests": 100,
    "window": 60,
    "burst": 20
  }
}
```

### Discovery Endpoints

- `GET /.well-known/agent-card` - Server agent card
- `GET /a2a/agents/{name}/card` - Specific agent card

## URL Structure

### Base Server Endpoints

- `GET /.well-known/agent-card` - Server capabilities
- `GET /a2a/health` - Health check
- `POST /a2a` - Default agent communication

### Agent-Specific Endpoints

- `GET /a2a/agents` - List available agents
- `GET /a2a/agents/{name}` - Agent information
- `GET /a2a/agents/{name}/card` - Agent capabilities
- `POST /a2a/agents/{name}` - Direct agent communication

## State Management

### Context Continuity

Conversations maintain state through consistent `contextId` usage:

```json
{
  "contextId": "user-123-session-456",
  "messages": [
    {
      "messageId": "msg-1",
      "contextId": "user-123-session-456",
      "role": "user",
      "parts": [{"kind": "text", "text": "Hello"}]
    },
    {
      "messageId": "msg-2", 
      "contextId": "user-123-session-456",
      "role": "agent",
      "parts": [{"kind": "text", "text": "Hi there!"}]
    }
  ]
}
```

### Task Lifecycle

```
[Client] ──message/send──→ [Server]
                             │
                             ▼
                         [Create Task]
                             │
                             ▼
                        [Status: submitted]
                             │
                             ▼
                        [Status: working]
                             │
                             ▼
                    [Status: completed/failed]
```

## Error Handling

### Standard Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse Error | Invalid JSON |
| -32600 | Invalid Request | Invalid JSON-RPC request |
| -32601 | Method Not Found | Unknown method |
| -32602 | Invalid Params | Invalid parameters |
| -32603 | Internal Error | Server internal error |
| -32000 | Agent Not Found | Specified agent not available |
| -32001 | Task Not Found | Specified task not found |
| -32002 | Agent Unavailable | Agent temporarily unavailable |
| -32003 | Rate Limited | Request rate exceeded |
| -32004 | Authentication Required | Authentication missing |
| -32005 | Permission Denied | Insufficient permissions |
| -32006 | Timeout | Request processing timeout |
| -32007 | Resource Exhausted | Server resources exhausted |

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "error": {
    "code": -32000,
    "message": "Agent not found",
    "data": {
      "agentName": "NonExistentAgent",
      "availableAgents": ["MathTutor", "ChatBot"],
      "timestamp": "2024-03-15T14:30:00Z",
      "requestId": "request-id",
      "traceId": "trace-12345"
    }
  }
}
```

## Security Considerations

### Authentication

Support for multiple authentication schemes:

```http
# Bearer token
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# API Key
X-API-Key: your-api-key-here

# Basic authentication
Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ=
```

### Rate Limiting

Rate limit headers in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 60
```

### Content Security

- Validate all input data
- Sanitize text content
- Limit message and part sizes
- Implement request timeouts

## Streaming Protocol

### Server-Sent Events Format

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Access-Control-Allow-Origin: *

event: status-update
data: {"jsonrpc":"2.0","id":"req-1","result":{"kind":"status-update","taskId":"task-123","status":{"state":"working","timestamp":"2024-03-15T14:30:01Z"},"final":false}}

event: partial-completion
data: {"jsonrpc":"2.0","id":"req-1","result":{"kind":"partial-completion","taskId":"task-123","message":{"role":"agent","parts":[{"kind":"text","text":"Partial response..."}],"messageId":"resp-1","contextId":"ctx-1","kind":"message"},"final":false}}

event: completion
data: {"jsonrpc":"2.0","id":"req-1","result":{"kind":"completion","taskId":"task-123","message":{"role":"agent","parts":[{"kind":"text","text":"Complete response here"}],"messageId":"resp-1","contextId":"ctx-1","kind":"message"},"final":true}}
```

### Event Types

- `status-update`: Task state change
- `partial-completion`: Incremental response
- `completion`: Final response
- `error`: Error occurred

## Performance Guidelines

### Message Limits

- Maximum message size: 1MB
- Maximum parts per message: 100
- Maximum text part size: 100KB
- Maximum data part size: 1MB

### Request Timeouts

- Default timeout: 30 seconds
- Maximum timeout: 300 seconds
- Streaming timeout: 600 seconds

### Concurrency Limits

- Maximum concurrent requests per client: 10
- Maximum concurrent tasks per context: 5
- Maximum context lifetime: 24 hours

## Protocol Extensions

### Custom Headers

Implementations may support custom headers for extended functionality:

```http
X-A2A-Version: 0.3.0
X-A2A-Client: jaf-python/2.0.0
X-A2A-Trace-Id: trace-12345
X-A2A-Priority: high
X-A2A-Timeout: 60000
```

### Metadata Support

Extended message format with metadata:

```json
{
  "role": "user",
  "parts": [{"kind": "text", "text": "Hello"}],
  "messageId": "msg-1",
  "contextId": "ctx-1",
  "kind": "message",
  "metadata": {
    "priority": "high",
    "tags": ["urgent", "customer-facing"],
    "userId": "user-123",
    "sessionInfo": {
      "userAgent": "Mozilla/5.0...",
      "ipAddress": "192.168.1.1"
    }
  }
}
```

## Compliance and Testing

### Protocol Compliance

Implementations must:
1. Support all core methods (`message/send`, `tasks/get`)
2. Handle all standard error codes appropriately
3. Maintain context consistency
4. Implement proper timeout handling
5. Support agent card discovery

### Testing Requirements

Test suites should verify:
- Request/response format compliance
- Error handling behavior
- Streaming functionality
- Authentication mechanisms
- Rate limiting enforcement

### Interoperability

Compliant implementations should be able to:
- Communicate with any A2A server
- Handle unknown message parts gracefully
- Degrade gracefully when features unavailable
- Maintain conversation state consistency

## Related Documentation

- [A2A Protocol Overview](a2a-protocol.md)
- [A2A API Reference](a2a-api-reference.md)
- [A2A Deployment Guide](a2a-deployment.md)
- [A2A Examples](a2a-examples.md)