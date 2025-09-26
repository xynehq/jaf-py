# JAF Human-in-the-Loop (HITL) Demo

This directory contains a comprehensive demo showcasing JAF's Human-in-the-Loop capability with file system operations in Python.

## File System HITL Demo

- **Safe Operations**: `listFiles`, `readFile` (no approval needed)
- **Dangerous Operations**: `deleteFile`, `editFile` (require approval)
- **Memory Provider Integration**: Uses environment-configured memory providers (Redis/PostgreSQL/in-memory)
- **Persistent Approval Storage**: Approval decisions stored in memory provider (survives restarts)
- **Conversation Continuity**: Resume interrupted conversations from exact point with approval context
- **Complete Status Tracking**: Full tool execution lifecycle (`halted` → `approved_and_executed`/`approval_denied`)
- **LLM Isolation**: Approval workflow invisible to LLM - no hallucinations
- **Audit Trail**: Complete approval history with timestamps and context
- **Async Pattern**: Native Python async/await throughout
- **Sandboxed Environment**: Secure file operations within demo directory

## 🚀 Quick Start

### Prerequisites

1. **JAF Root Directory**: Make sure you're in the JAF root directory:
```bash
cd /path/to/jaf-py
```

2. **Python Dependencies**: Install required dependencies:
```bash
pip install -e .
pip install fastapi uvicorn  # For API demo
```

3. **Model Provider Configuration**: 
```bash
# Copy and configure environment
cp examples/hitl-demo/.env.example examples/hitl-demo/.env
# Edit .env with your model provider settings
```

Example `.env` (choose one option):
```bash
# Option 1: LiteLLM (recommended - supports multiple providers)
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=sk-demo
LITELLM_MODEL=gpt-4o-mini
```

### Run the Demo

#### 🗂️ Interactive File System HITL Demo
```bash
python examples/hitl-demo/demo.py
```

This runs the interactive file system demo where you can:
- Chat with the AI assistant about file operations
- Perform safe operations (list, read) immediately
- Get approval prompts for dangerous operations (delete, edit)
- See approval context flow to tool execution
- Experience persistent approval storage across sessions

#### 🌐 API Demo with HTTP Endpoints
```bash
python examples/hitl-demo/api_demo.py
```

This runs both terminal interaction AND HTTP endpoints for approvals.

## 🌐 API Demo Usage

When running `api_demo.py`, you get both terminal interaction AND HTTP endpoints:

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/pending` | List all pending tool approvals |
| `POST` | `/approve/{sessionId}/{toolCallId}` | Approve a specific tool call |
| `POST` | `/reject/{sessionId}/{toolCallId}` | Reject a specific tool call |
| `GET` | `/health` | Health check and pending count |
| `GET` | `/approvals/stream?conversationId=...` | SSE stream for real-time updates |

### Example Workflow

1. **Start the API demo:**
   ```bash
   python examples/hitl-demo/api_demo.py
   ```

2. **Check pending approvals via curl:**
   ```bash
   curl http://localhost:3001/pending
   ```

3. **Approve via curl (simple):**
   ```bash
   curl -X POST http://localhost:3001/approve/SESSION_ID/TOOL_CALL_ID
   ```

4. **Approve with additional context:**
   ```bash
   curl -X POST http://localhost:3001/approve/SESSION_ID/TOOL_CALL_ID \
        -H "Content-Type: application/json" \
        -d '{
          "additionalContext": {
            "message": "your-additional-context"
          }
        }'
   ```

5. **Approve with image context (base64 data):**
   ```bash
   curl -X POST http://localhost:3001/approve/SESSION_ID/TOOL_CALL_ID \
        -H "Content-Type: application/json" \
        -d '{
          "additionalContext": {
            "messages": [
              {
                "role": "user",
                "content": "Analyze this image and make your decision based on it",
                "attachments": [
                  {
                    "kind": "image",
                    "mime_type": "image/png",
                    "name": "test-pixel.png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                  }
                ]
              }
            ]
          }
        }'
   ```

6. **Approve with image context (URL):**
   ```bash
   curl -X POST http://localhost:3001/approve/SESSION_ID/TOOL_CALL_ID \
        -H "Content-Type: application/json" \
        -d '{
          "additionalContext": {
            "messages": [
              {
                "role": "user",
                "content": "Here is the image for context",
                "attachments": [
                  {
                    "kind": "image",
                    "mime_type": "image/jpeg",
                    "name": "example.jpg",
                    "url": "https://fastly.picsum.photos/id/21/1920/1080.jpg?hmac=1BnxKswnhchVaU4-xZpgObgnwGLLb7hnugRQ9vwwUFY"
                  }
                ]
              }
            ]
          }
        }'
   ```

7. **Reject via curl (simple):**
   ```bash
   curl -X POST http://localhost:3001/reject/SESSION_ID/TOOL_CALL_ID
   ```

8. **Reject with additional context:**
   ```bash
   curl -X POST http://localhost:3001/reject/SESSION_ID/TOOL_CALL_ID \
        -H "Content-Type: application/json" \
        -d '{
          "reason": "not authorized",
          "additionalContext": {
            "rejectedBy": "your-name"
          }
        }'
   ```

### Image Context Support

The approval API now supports **image attachments** as additional context! This allows users to provide visual context when approving/rejecting tool calls.

**Supported image formats:**
- **Base64 data**: Include image data directly in the request
- **Remote URLs**: Reference images hosted elsewhere
- **MIME types**: `image/png`, `image/jpeg`, `image/gif`, `image/webp`, etc.

**How it works:**
1. Include images in `additionalContext.messages[].attachments`
2. JAF automatically processes and adds images to the conversation
3. Vision-capable LLMs (GPT-4V, Claude 3, Gemini Pro Vision) can analyze the images
4. The LLM makes informed decisions based on visual context

**Example use cases:**
- Approve file deletion after reviewing a screenshot
- Validate changes by showing before/after images
- Provide visual instructions or clarifications
- Share error screenshots for better context

### Configuration

Additional API demo configuration in `.env`:
```bash
API_PORT=3001  # Port for HTTP API server
```

## 🔧 Memory Provider Configuration

The demo supports different memory providers:

### In-Memory (Default)
```bash
JAF_MEMORY_TYPE=memory
```

### Redis
```bash
JAF_MEMORY_TYPE=redis
JAF_REDIS_HOST=localhost
JAF_REDIS_PORT=6379
JAF_REDIS_DB=0
```

### PostgreSQL
```bash
JAF_MEMORY_TYPE=postgres
JAF_POSTGRES_HOST=localhost
JAF_POSTGRES_PORT=5432
JAF_POSTGRES_DB=jaf_memory
JAF_POSTGRES_USER=postgres
JAF_POSTGRES_PASSWORD=your_password_here
JAF_POSTGRES_SSL=false
```


## 📁 Demo Structure

```
examples/hitl-demo/
├── README.md                 # This file
├── .env.example             # Environment configuration template
├── demo.py                  # Interactive terminal demo
├── api_demo.py              # HTTP API demo with curl support
├── shared/
│   ├── agent.py            # File system agent definition
│   ├── tools.py            # File system tools with approval logic
│   └── memory.py           # Memory provider configuration
└── sandbox/                # Sandboxed directory for file operations
    ├── README.txt          # Sample file
    ├── config.json         # Sample configuration file
    └── notes.md            # Sample markdown file
```