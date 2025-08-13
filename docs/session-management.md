# Session Management

!!! info "Immutable Sessions"
    JAF implements immutable session management following functional programming principles. All session operations create new sessions rather than modifying existing ones, ensuring thread safety and predictable behavior.

## 🎯 Overview

JAF's session management system provides:

- **🔒 Immutable Data Structures**: Sessions never change after creation
- **🧠 Pure Functions**: All operations are side-effect free
- **⚡ Thread Safety**: Concurrent access is safe by design
- **🔄 Functional Composition**: Build complex workflows by composing simple operations

## 🏗️ Core Concepts

### Immutable Session Architecture

```mermaid
graph TD
    A[Original Session] --> B[with_message()]
    B --> C[New Session + Message]
    A --> D[with_metadata()]
    D --> E[New Session + Metadata]
    A --> F[get_recent_messages()]
    F --> G[Message List]
    
    style A fill:#e1f5fe
    style C fill:#c8e6c9
    style E fill:#c8e6c9
    style G fill:#fff3e0
```

### Before vs After: Session Management

#### Before: Mutable Sessions (Prototype)
```python
# ❌ Old approach - mutable state, not thread-safe
class OldSession:
    def __init__(self, session_id):
        self.messages = []  # Mutable list
        self.metadata = {}  # Mutable dict
    
    def add_message(self, message):
        self.messages.append(message)  # Modifies existing session
        return self  # Returns same object
```

#### After: Immutable Sessions (Production)
```python
# ✅ New approach - immutable, thread-safe
@dataclass(frozen=True)
class ImmutableAdkSession:
    messages: Tuple[AdkMessage, ...]  # Immutable tuple
    metadata: FrozenDict[str, Any]    # Immutable mapping
    
    def with_message(self, message: AdkMessage) -> 'ImmutableAdkSession':
        return ImmutableAdkSession(
            messages=self.messages + (message,),  # Creates new tuple
            metadata=self.metadata,               # Reuses immutable data
            # ... other fields
        )
```

## 🔧 Creating Sessions

### Basic Session Creation

```python
from adk.types import create_immutable_session

# Create a new immutable session
session = create_immutable_session(
    session_id="user-123-session",
    user_id="user-123",
    app_name="my-agent-app"
)

print(f"Session ID: {session.session_id}")
print(f"User ID: {session.user_id}")
print(f"Messages: {len(session.messages)}")  # 0 - starts empty
```

### Session with Initial Data

```python
from adk.types import create_immutable_session, create_user_message
from datetime import datetime

# Create session with metadata
session = create_immutable_session(
    session_id="advanced-session",
    user_id="user-456", 
    app_name="advanced-app",
    created_at=datetime.now(),
    metadata={
        "user_preferences": {"theme": "dark", "language": "en"},
        "session_type": "conversation",
        "priority": "high"
    }
)
```

## 💬 Managing Messages

### Adding Messages Functionally

```python
from adk.types import create_user_message, create_assistant_message

# Start with empty session
session = create_immutable_session("demo", "user", "app")

# Add user message (creates new session)
user_msg = create_user_message("Hello, how can you help me?")
session_with_user_msg = session.with_message(user_msg)

# Add assistant response (creates another new session)
assistant_msg = create_assistant_message("I can help you with various tasks!")
session_with_response = session_with_user_msg.with_message(assistant_msg)

# Original session is unchanged
print(f"Original: {len(session.messages)} messages")              # 0
print(f"With user: {len(session_with_user_msg.messages)} messages")  # 1
print(f"With response: {len(session_with_response.messages)} messages")  # 2
```

### Building Conversations

```python
# Functional conversation building
session = create_immutable_session("conversation", "user", "app")

# Chain operations functionally
conversation = (session
    .with_message(create_user_message("What's the weather like?"))
    .with_message(create_assistant_message("I'd need your location to check the weather."))
    .with_message(create_user_message("I'm in San Francisco"))
    .with_message(create_assistant_message("It's currently 72°F and sunny in San Francisco!"))
)

print(f"Complete conversation: {len(conversation.messages)} messages")
```

### Message Types

```python
from adk.types import create_system_message, create_tool_message

# Different message types
system_msg = create_system_message("You are a helpful AI assistant")
user_msg = create_user_message("Calculate 15 * 7")
tool_msg = create_tool_message("calculator", {"result": 105})
assistant_msg = create_assistant_message("15 * 7 equals 105")

# Build session with all message types
full_session = (create_immutable_session("calc", "user", "app")
    .with_message(system_msg)
    .with_message(user_msg)
    .with_message(tool_msg)
    .with_message(assistant_msg)
)
```

## 🔍 Querying Sessions

### Retrieving Recent Messages

```python
# Get recent messages (pure function)
recent_messages = session.get_recent_messages(count=5)
print(f"Last 5 messages: {len(recent_messages)}")

# Get messages by role
user_messages = session.get_messages_by_role("user")
assistant_messages = session.get_messages_by_role("assistant")
```

### Message Filtering

```python
from datetime import datetime, timedelta

# Get messages from last hour
one_hour_ago = datetime.now() - timedelta(hours=1)
recent_msgs = session.get_messages_after(one_hour_ago)

# Get messages containing specific text
search_results = session.search_messages("weather")
```

### Session Statistics

```python
# Get session statistics (pure functions)
stats = session.get_statistics()
print(f"Total messages: {stats['total_messages']}")
print(f"User messages: {stats['user_messages']}")
print(f"Assistant messages: {stats['assistant_messages']}")
print(f"Session duration: {stats['duration_minutes']} minutes")
```

## 🔄 Pure Function Operations

### Functional Session Operations

```python
from adk.types import (
    add_message_to_session,
    add_metadata_to_session,
    filter_messages_by_role,
    merge_sessions
)

# Pure function: add message
original_session = create_immutable_session("pure", "user", "app")
message = create_user_message("Test message")

new_session = add_message_to_session(original_session, message)

# Original unchanged
assert len(original_session.messages) == 0
assert len(new_session.messages) == 1

# Pure function: add metadata
session_with_metadata = add_metadata_to_session(
    original_session, 
    {"experiment": "A/B test", "version": "1.2.0"}
)

# Pure function: filter messages
user_messages = filter_messages_by_role(new_session, "user")
```

### Session Transformation Pipeline

```python
from adk.types import transform_session

# Create transformation pipeline
def add_system_context(session):
    """Add system context to session."""
    system_msg = create_system_message("You are in helpful mode")
    return session.with_message(system_msg)

def add_user_greeting(session):
    """Add user greeting."""
    greeting = create_user_message("Hello!")
    return session.with_message(greeting)

def add_assistant_response(session):
    """Add assistant response."""
    response = create_assistant_message("Hello! How can I help you?")
    return session.with_message(response)

# Transform session through pipeline
empty_session = create_immutable_session("pipeline", "user", "app")

complete_session = transform_session(
    empty_session,
    transformations=[
        add_system_context,
        add_user_greeting, 
        add_assistant_response
    ]
)

print(f"Pipeline result: {len(complete_session.messages)} messages")
```

## 🔒 Thread Safety

### Concurrent Operations

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor

def concurrent_message_addition(base_session, thread_id, results):
    """Add messages concurrently."""
    current_session = base_session
    
    for i in range(10):
        message = create_user_message(f"Thread {thread_id} message {i}")
        current_session = current_session.with_message(message)
        time.sleep(0.001)  # Simulate processing time
    
    results[thread_id] = current_session

# Base session shared across threads
base_session = create_immutable_session("concurrent", "user", "app")
results = {}

# Run concurrent operations
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for i in range(5):
        future = executor.submit(concurrent_message_addition, base_session, i, results)
        futures.append(future)
    
    # Wait for all threads to complete
    for future in futures:
        future.result()

# Each thread produced independent results
for thread_id, session in results.items():
    print(f"Thread {thread_id}: {len(session.messages)} messages")

# Base session remains unchanged
print(f"Base session: {len(base_session.messages)} messages")  # Still 0
```

### Race Condition Prevention

```python
# Immutable sessions prevent race conditions
shared_session = create_immutable_session("shared", "user", "app")

def safe_concurrent_access(session, operation_id):
    """Safely access session concurrently."""
    # Reading is always safe - immutable data
    message_count = len(session.messages)
    session_id = session.session_id
    
    # Creating new sessions is safe - no shared mutable state
    new_message = create_user_message(f"Operation {operation_id}")
    new_session = session.with_message(new_message)
    
    return new_session

# Multiple threads can safely read and create new sessions
# No locks or synchronization needed
```

## 💾 Session Persistence

### Session Providers

```python
from adk.sessions import create_in_memory_session_provider, create_redis_session_provider

# In-memory provider for development
memory_provider = create_in_memory_session_provider({
    "max_sessions": 1000,
    "ttl_seconds": 3600
})

# Redis provider for production
redis_provider = create_redis_session_provider({
    "url": "redis://localhost:6379",
    "max_connections": 10,
    "key_prefix": "jaf:session:"
})
```

### Storing and Retrieving Sessions

```python
# Store session
session = create_immutable_session("persistent", "user", "app")
session_with_data = session.with_message(create_user_message("Hello"))

store_result = await redis_provider.store_session(session_with_data)
if store_result.success:
    print("Session stored successfully")

# Retrieve session
retrieve_result = await redis_provider.get_session("persistent")
if retrieve_result.success:
    retrieved_session = retrieve_result.session
    print(f"Retrieved {len(retrieved_session.messages)} messages")
```

### Session Serialization

```python
from adk.types import serialize_session, deserialize_session

# Serialize session to JSON
session_json = serialize_session(session_with_data)
print(f"Serialized size: {len(session_json)} bytes")

# Deserialize back to session
restored_session = deserialize_session(session_json)
assert restored_session.session_id == session_with_data.session_id
assert len(restored_session.messages) == len(session_with_data.messages)
```

## 🧪 Testing Session Management

### Unit Tests for Immutability

```python
def test_session_immutability():
    """Test that sessions are truly immutable."""
    original = create_immutable_session("test", "user", "app")
    message = create_user_message("Test")
    
    # Adding message creates new session
    modified = original.with_message(message)
    
    # Original is unchanged
    assert len(original.messages) == 0
    assert len(modified.messages) == 1
    assert original != modified
    assert original.session_id == modified.session_id

def test_pure_function_behavior():
    """Test that session functions are pure."""
    session = create_immutable_session("pure", "user", "app")
    message = create_user_message("Pure test")
    
    # Multiple calls with same inputs produce same outputs
    result1 = add_message_to_session(session, message)
    result2 = add_message_to_session(session, message)
    
    assert result1.messages == result2.messages
    assert result1 != session  # New object created
    assert result2 != session  # New object created
```

### Performance Tests

```python
import time

def test_session_performance():
    """Test session creation and manipulation performance."""
    start_time = time.time()
    
    # Create base session
    session = create_immutable_session("perf", "user", "app")
    
    # Add 1000 messages
    for i in range(1000):
        message = create_user_message(f"Message {i}")
        session = session.with_message(message)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Added 1000 messages in {duration:.3f} seconds")
    print(f"Rate: {1000/duration:.0f} messages/second")
    
    assert len(session.messages) == 1000
    assert duration < 1.0  # Should be fast
```

## 📊 Best Practices

### 1. Session Design Patterns

#### Builder Pattern
```python
class SessionBuilder:
    """Build sessions step by step."""
    
    def __init__(self, session_id: str, user_id: str, app_name: str):
        self._session = create_immutable_session(session_id, user_id, app_name)
    
    def with_system_context(self, context: str) -> 'SessionBuilder':
        msg = create_system_message(context)
        self._session = self._session.with_message(msg)
        return self
    
    def with_user_input(self, input_text: str) -> 'SessionBuilder':
        msg = create_user_message(input_text)
        self._session = self._session.with_message(msg)
        return self
    
    def build(self) -> ImmutableAdkSession:
        return self._session

# Usage
session = (SessionBuilder("builder", "user", "app")
    .with_system_context("You are a helpful assistant")
    .with_user_input("Hello!")
    .build())
```

#### Session Factory
```python
def create_conversation_session(user_id: str, context: str = None) -> ImmutableAdkSession:
    """Factory for conversation sessions."""
    session_id = f"{user_id}-{int(time.time())}"
    session = create_immutable_session(session_id, user_id, "conversation")
    
    if context:
        system_msg = create_system_message(context)
        session = session.with_message(system_msg)
    
    return session

# Usage
session = create_conversation_session("user-123", "Math tutor mode")
```

### 2. Memory Management

```python
# Keep sessions lightweight
def cleanup_old_messages(session: ImmutableAdkSession, max_messages: int = 100) -> ImmutableAdkSession:
    """Keep only recent messages to manage memory."""
    if len(session.messages) <= max_messages:
        return session
    
    recent_messages = session.messages[-max_messages:]
    return session._replace(messages=recent_messages)

# Usage
large_session = session_with_many_messages
cleaned_session = cleanup_old_messages(large_session, max_messages=50)
```

### 3. Error Handling

```python
from adk.types import SessionError

def safe_add_message(session: ImmutableAdkSession, message: AdkMessage) -> ImmutableAdkSession:
    """Safely add message with validation."""
    try:
        # Validate message
        if not message.content.strip():
            raise SessionError("Message content cannot be empty")
        
        # Add message
        return session.with_message(message)
    
    except Exception as e:
        # Log error and return original session
        logger.error(f"Failed to add message: {e}")
        return session
```

## 🔗 Related Documentation

- **[ADK Overview](adk-overview.md)** - Complete ADK framework introduction
- **[Security Framework](security-framework.md)** - Security and session protection
- **[Error Handling](error-handling.md)** - Robust error recovery patterns
- **[Validation Suite](validation-suite.md)** - Testing session management

---

!!! success "Functional Sessions"
    JAF's immutable session management provides thread-safe, predictable behavior through functional programming principles. The transformation from mutable to immutable sessions eliminated race conditions and improved system reliability.