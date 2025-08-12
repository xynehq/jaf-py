# Memory System

JAF provides a robust conversation memory system that enables persistent conversations across sessions. The memory system supports multiple backends and provides a clean abstraction for storing and retrieving conversation history.

## Overview

The memory system in JAF is designed with several key principles:

- **Provider Abstraction**: Use any backend (in-memory, Redis, PostgreSQL) with the same interface
- **Type Safety**: Full Python type hints and Pydantic validation
- **Functional Design**: Immutable data structures and result types
- **Environment Configuration**: Easy setup through environment variables
- **Automatic Integration**: Seamless integration with the JAF engine

## Core Concepts

### ConversationMemory

The `ConversationMemory` dataclass represents a complete conversation:

```python
from jaf.memory import ConversationMemory
from jaf.core.types import Message

# Immutable conversation object
conversation = ConversationMemory(
    conversation_id="user-123-session-1",
    user_id="user-123", 
    messages=[
        Message(role="user", content="Hello!"),
        Message(role="assistant", content="Hi there! How can I help you?")
    ],
    metadata={"session_start": "2024-01-15T10:00:00Z"}
)
```

### MemoryProvider Protocol

All memory providers implement the `MemoryProvider` protocol:

```python
from jaf.memory import MemoryProvider, MemoryQuery, ConversationMemory
from typing import List, Optional, Dict, Any

class MyCustomProvider:
    async def store_messages(
        self, 
        conversation_id: str, 
        messages: List[Message],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Store messages for a conversation."""
        
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Retrieve complete conversation history."""
        
    async def append_messages(
        self,
        conversation_id: str,
        messages: List[Message], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result:
        """Add new messages to existing conversation."""
        
    async def get_recent_messages(
        self, 
        conversation_id: str, 
        limit: int = 50
    ) -> List[Message]:
        """Get recent messages from conversation."""
        
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and return success status."""
        
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health and connectivity."""
```

## Available Providers

### In-Memory Provider

Perfect for development and testing. Conversations are lost when the application restarts.

```python
from jaf.memory import create_in_memory_provider, InMemoryConfig

# Create provider with configuration
config = InMemoryConfig(
    max_conversations=1000,  # Maximum conversations to store
    max_messages=1000        # Maximum messages per conversation
)

provider = create_in_memory_provider(config)
```

**Environment Variables:**
```bash
JAF_MEMORY_TYPE=memory
JAF_MEMORY_MAX_CONVERSATIONS=1000
JAF_MEMORY_MAX_MESSAGES=1000
```

**Characteristics:**
- ✅ No external dependencies
- ✅ Instant setup
- ✅ Perfect for development
- ❌ Data lost on restart
- ❌ No persistence
- ❌ Limited by RAM

### Redis Provider

High-performance, in-memory storage with optional persistence.

```python
from jaf.memory import create_redis_provider, RedisConfig
import redis.asyncio as redis

# Method 1: Create with config and client
redis_client = redis.Redis(host="localhost", port=6379, db=0)
config = RedisConfig(
    host="localhost",
    port=6379,
    db=0,
    key_prefix="jaf:memory:",
    ttl=86400  # 24 hours
)

provider = await create_redis_provider(config, redis_client)

# Method 2: Create from URL
config = RedisConfig(url="redis://localhost:6379/0")
provider = await create_redis_provider(config)
```

**Environment Variables:**
```bash
JAF_MEMORY_TYPE=redis

# Option 1: Full URL
JAF_REDIS_URL=redis://localhost:6379/0

# Option 2: Individual parameters  
JAF_REDIS_HOST=localhost
JAF_REDIS_PORT=6379
JAF_REDIS_PASSWORD=your-password
JAF_REDIS_DB=0
JAF_REDIS_KEY_PREFIX=jaf:memory:
JAF_REDIS_TTL=86400
```

**Installation:**
```bash
pip install redis
```

**Characteristics:**
- ✅ High performance
- ✅ Horizontal scaling
- ✅ Optional persistence
- ✅ TTL support
- ✅ Production ready
- ⚠️ Requires Redis server

### PostgreSQL Provider

Robust, ACID-compliant relational database storage.

```python
from jaf.memory import create_postgres_provider, PostgresConfig
import asyncpg

# Method 1: Create with config and connection
connection = await asyncpg.connect("postgresql://user:pass@localhost/jaf_memory")
config = PostgresConfig(
    host="localhost",
    port=5432,
    database="jaf_memory",
    username="postgres",
    password="your-password",
    table_name="conversations"
)

provider = await create_postgres_provider(config, connection)

# Method 2: Create from connection string
config = PostgresConfig(
    connection_string="postgresql://user:pass@localhost/jaf_memory"
)
provider = await create_postgres_provider(config)
```

**Environment Variables:**
```bash
JAF_MEMORY_TYPE=postgres

# Option 1: Connection string
JAF_POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost/jaf_memory

# Option 2: Individual parameters
JAF_POSTGRES_HOST=localhost
JAF_POSTGRES_PORT=5432
JAF_POSTGRES_DATABASE=jaf_memory
JAF_POSTGRES_USERNAME=postgres
JAF_POSTGRES_PASSWORD=your-password
JAF_POSTGRES_SSL=false
JAF_POSTGRES_TABLE_NAME=conversations
JAF_POSTGRES_MAX_CONNECTIONS=10
```

**Installation:**
```bash
pip install asyncpg
```

**Database Schema:**
```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    messages JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);
```

**Characteristics:**
- ✅ ACID transactions
- ✅ Complex queries
- ✅ Strong consistency
- ✅ Backup/restore
- ✅ Enterprise ready
- ⚠️ Requires PostgreSQL server

## Environment-Based Configuration

JAF provides automatic provider creation from environment variables:

```python
from jaf.memory import create_memory_provider_from_env, MemoryConfig

# Create provider based on JAF_MEMORY_TYPE
provider = await create_memory_provider_from_env()

# Create memory config for engine
memory_config = MemoryConfig(
    provider=provider,
    auto_store=True,      # Automatically store conversations
    max_messages=1000,    # Limit messages per conversation
    ttl=86400            # Time to live in seconds
)
```

### Provider Info and Testing

```python
from jaf.memory import get_memory_provider_info, test_memory_provider_connection

# Get configuration info without creating provider
info = get_memory_provider_info()
print(f"Provider type: {info['type']}")
print(f"Persistence: {info['persistence']}")

# Test connection before creating provider
result = await test_memory_provider_connection()
if result['healthy']:
    print(f"✅ {result['message']}")
else:
    print(f"❌ {result['error']}")
```

## Integration with JAF Engine

### Automatic Memory Integration

```python
from jaf import run, RunState, RunConfig, Message, Agent
from jaf.memory import create_memory_provider_from_env, MemoryConfig

# Create memory provider
memory_provider = await create_memory_provider_from_env()
memory_config = MemoryConfig(
    provider=memory_provider,
    auto_store=True,
    max_messages=100
)

# Create agent
agent = Agent(
    name="ChatBot",
    instructions=lambda state: "You are a helpful assistant.",
    tools=[]
)

# Run with memory
initial_state = RunState(
    messages=[Message(role="user", content="Hello!")],
    current_agent_name="ChatBot",
    context={"user_id": "user-123"}
)

config = RunConfig(
    agent_registry={"ChatBot": agent},
    model_provider=your_model_provider,
    memory=memory_config,
    conversation_id="user-123-session-1"  # Important: specify conversation ID
)

result = await run(initial_state, config)
```

### Manual Memory Operations

```python
# Store conversation manually
from jaf.memory import ConversationMemory

messages = [
    Message(role="user", content="What's the weather like?"),
    Message(role="assistant", content="I'd need your location to check the weather.")
]

# Store new conversation
result = await provider.store_messages(
    conversation_id="weather-chat-1",
    messages=messages,
    metadata={"topic": "weather", "user_location": "unknown"}
)

# Append to existing conversation
new_messages = [
    Message(role="user", content="I'm in New York"),
    Message(role="assistant", content="It's currently 72°F and sunny in New York!")
]

result = await provider.append_messages(
    conversation_id="weather-chat-1",
    messages=new_messages
)

# Retrieve conversation
conversation = await provider.get_conversation("weather-chat-1")
if conversation:
    print(f"Found {len(conversation.messages)} messages")
    for message in conversation.messages:
        print(f"{message.role}: {message.content}")
```

## Advanced Usage

### Conversation Search and Management

```python
from jaf.memory import MemoryQuery
from datetime import datetime, timedelta

# Find conversations for a user
query = MemoryQuery(
    user_id="user-123",
    limit=10,
    since=datetime.now() - timedelta(days=7)  # Last 7 days
)

conversations = await provider.find_conversations(query)
for conv in conversations:
    print(f"Conversation {conv.conversation_id}: {len(conv.messages)} messages")

# Get recent messages only
recent_messages = await provider.get_recent_messages(
    conversation_id="user-123-session-1",
    limit=20
)

# Get conversation statistics
stats = await provider.get_stats(user_id="user-123")
print(f"Total conversations: {stats['total_conversations']}")
print(f"Total messages: {stats['total_messages']}")

# Clear user data (GDPR compliance)
deleted_count = await provider.clear_user_conversations("user-123")
print(f"Deleted {deleted_count} conversations")
```

### Custom Metadata and Context

```python
# Store rich metadata with conversations
metadata = {
    "session_info": {
        "user_agent": "Mozilla/5.0...",
        "ip_address": "192.168.1.1",
        "session_start": "2024-01-15T10:00:00Z"
    },
    "conversation_context": {
        "topic": "customer_support",
        "priority": "high",
        "department": "billing"
    },
    "user_preferences": {
        "language": "en",
        "timezone": "America/New_York",
        "notification_settings": {"email": True, "sms": False}
    }
}

await provider.store_messages(
    conversation_id="support-ticket-456",
    messages=messages,
    metadata=metadata
)
```

### Error Handling

```python
from jaf.memory import (
    MemoryError, MemoryConnectionError, 
    MemoryNotFoundError, MemoryStorageError,
    Success, Failure
)

try:
    result = await provider.store_messages(conversation_id, messages)
    
    # Check result type (functional error handling)
    if isinstance(result, Success):
        print("Messages stored successfully")
    elif isinstance(result, Failure):
        print(f"Storage failed: {result.error}")
        
except MemoryConnectionError as e:
    print(f"Connection failed to {e.provider}: {e}")
    # Implement fallback or retry logic
    
except MemoryStorageError as e:
    print(f"Storage operation '{e.operation}' failed: {e}")
    # Log error and potentially use fallback storage
    
except MemoryNotFoundError as e:
    print(f"Conversation {e.conversation_id} not found")
    # Handle missing conversation scenario
```

## Production Configuration

### Redis Production Setup

```bash
# High-availability Redis with persistence
JAF_MEMORY_TYPE=redis
JAF_REDIS_URL=redis://auth-token@redis-cluster.company.com:6380/0
JAF_REDIS_KEY_PREFIX=prod:jaf:memory:
JAF_REDIS_TTL=2592000  # 30 days

# Optional: Redis Sentinel for HA
JAF_REDIS_SENTINEL_HOSTS=sentinel1:26379,sentinel2:26379,sentinel3:26379
JAF_REDIS_SENTINEL_SERVICE_NAME=mymaster
```

### PostgreSQL Production Setup

```bash
# Production PostgreSQL with SSL
JAF_MEMORY_TYPE=postgres
JAF_POSTGRES_CONNECTION_STRING=postgresql://jaf_user:secure_password@postgres.company.com:5432/jaf_production?sslmode=require
JAF_POSTGRES_TABLE_NAME=prod_conversations
JAF_POSTGRES_MAX_CONNECTIONS=20
JAF_POSTGRES_SSL=true

# Connection pooling (recommended)
JAF_POSTGRES_POOL_MIN_SIZE=5
JAF_POSTGRES_POOL_MAX_SIZE=20
JAF_POSTGRES_POOL_MAX_QUERIES=50000
JAF_POSTGRES_POOL_MAX_INACTIVE_CONNECTION_LIFETIME=300
```

### Memory Configuration Optimization

```python
# Production memory configuration
memory_config = MemoryConfig(
    provider=provider,
    auto_store=True,
    max_messages=1000,           # Limit conversation length
    ttl=2592000,                # 30 days retention
    compression_threshold=100    # Compress conversations > 100 messages
)
```

## Monitoring and Observability

### Health Checks

```python
# Regular health monitoring
async def monitor_memory_health():
    health = await provider.health_check()
    
    if health.get('healthy'):
        print(f"✅ Memory provider healthy: {health.get('message')}")
        
        # Log performance metrics
        metrics = health.get('metrics', {})
        print(f"   - Connections: {metrics.get('active_connections', 'N/A')}")
        print(f"   - Memory usage: {metrics.get('memory_usage', 'N/A')}")
        print(f"   - Response time: {metrics.get('avg_response_time', 'N/A')}ms")
    else:
        print(f"❌ Memory provider unhealthy: {health.get('error')}")
        
        # Alert operations team
        await send_alert(f"Memory provider failure: {health.get('error')}")

# Schedule regular health checks
import asyncio
asyncio.create_task(monitor_memory_health())
```

### Performance Metrics

```python
# Track conversation statistics
stats = await provider.get_stats()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Total messages: {stats['total_messages']}")
print(f"Average messages per conversation: {stats['avg_messages_per_conversation']}")
print(f"Storage size: {stats['total_storage_size']} bytes")

# Per-user statistics
user_stats = await provider.get_stats(user_id="user-123")
print(f"User conversations: {user_stats['user_conversations']}")
print(f"User messages: {user_stats['user_messages']}")
```

## Best Practices

### 1. Conversation ID Strategy

```python
# Use structured conversation IDs
def create_conversation_id(user_id: str, session_type: str, timestamp: str) -> str:
    return f"{user_id}:{session_type}:{timestamp}"

# Examples:
# "user-123:chat:2024-01-15T10:00:00Z"
# "user-456:support:2024-01-15T14:30:00Z"
# "user-789:onboarding:2024-01-15T16:45:00Z"
```

### 2. Message Limits and Cleanup

```python
# Implement conversation cleanup
async def cleanup_old_conversations():
    cutoff_date = datetime.now() - timedelta(days=90)
    
    # Find old conversations
    query = MemoryQuery(until=cutoff_date, limit=1000)
    old_conversations = await provider.find_conversations(query)
    
    # Archive or delete
    for conv in old_conversations:
        if should_archive(conv):
            await archive_conversation(conv)
        await provider.delete_conversation(conv.conversation_id)
```

### 3. Data Privacy and Compliance

```python
# GDPR-compliant user data deletion
async def delete_user_data(user_id: str):
    # Get user consent verification
    if not verify_deletion_consent(user_id):
        raise ValueError("User deletion requires verified consent")
    
    # Delete all user conversations
    deleted_count = await provider.clear_user_conversations(user_id)
    
    # Log deletion for compliance
    audit_log.info(f"Deleted {deleted_count} conversations for user {user_id}")
    
    return deleted_count
```

### 4. Backup and Recovery

```python
# Export conversations for backup
async def export_conversations(user_id: Optional[str] = None) -> Dict:
    query = MemoryQuery(user_id=user_id, limit=None)
    conversations = await provider.find_conversations(query)
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "conversation_count": len(conversations),
        "conversations": [
            {
                "conversation_id": conv.conversation_id,
                "user_id": conv.user_id,
                "messages": [msg.dict() for msg in conv.messages],
                "metadata": conv.metadata
            }
            for conv in conversations
        ]
    }
    
    return export_data

# Import from backup
async def import_conversations(export_data: Dict):
    for conv_data in export_data["conversations"]:
        messages = [Message(**msg) for msg in conv_data["messages"]]
        
        await provider.store_messages(
            conversation_id=conv_data["conversation_id"],
            messages=messages,
            metadata=conv_data["metadata"]
        )
```

## Troubleshooting

### Common Issues

**1. Connection Failures**
```python
# Test connection independently
result = await test_memory_provider_connection()
if not result['healthy']:
    print(f"Connection issue: {result['error']}")
```

**2. Performance Issues**
```python
# Monitor response times
import time

start_time = time.time()
conversation = await provider.get_conversation("test-id")
response_time = (time.time() - start_time) * 1000

if response_time > 100:  # > 100ms
    print(f"Slow response: {response_time:.2f}ms")
```

**3. Memory Leaks**
```python
# Properly close providers
try:
    # Use provider
    pass
finally:
    await provider.close()
```

**4. Data Consistency**
```python
# Verify data integrity
async def verify_conversation_integrity(conversation_id: str):
    conversation = await provider.get_conversation(conversation_id)
    if not conversation:
        return False
    
    # Check message sequence
    for i, message in enumerate(conversation.messages):
        if not message.content or not message.role:
            print(f"Invalid message at index {i}")
            return False
    
    return True
```

## Next Steps

- Learn about [Model Providers](model-providers.md) for LLM integration
- Explore [Server API](server-api.md) for HTTP endpoints
- Check [Deployment](deployment.md) for production setup
- Review [Examples](examples.md) for real-world usage patterns