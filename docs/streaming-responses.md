# Streaming Responses

JAF's streaming response system enables real-time, progressive content delivery for enhanced user experiences. This system provides streaming capabilities for agent responses, tool calls, and execution metadata.

## Overview

The streaming system provides:

- **Real-time Content Delivery**: Stream responses as they're generated
- **Progressive Updates**: Receive chunks, tool calls, and metadata progressively
- **Event-based Architecture**: Handle different types of streaming events
- **Buffer Management**: Accumulate and manage streaming content
- **Integration Support**: Easy integration with WebSockets and SSE

## Core Components

### StreamingEvent

The main streaming unit that contains progressive updates:

```python
from jaf.core.streaming import StreamingEvent, StreamingEventType, run_streaming
from jaf.core.types import RunState, RunConfig

# Stream agent responses
async def stream_agent_response(initial_state: RunState, config: RunConfig):
    async for event in run_streaming(initial_state, config):
        print(f"Event type: {event.type}")
        print(f"Timestamp: {event.timestamp}")
        
        if event.type == StreamingEventType.CHUNK:
            chunk = event.data
            print(f"Content: {chunk.content}")
            print(f"Delta: {chunk.delta}")
            print(f"Complete: {chunk.is_complete}")
        
        elif event.type == StreamingEventType.TOOL_CALL:
            tool_call = event.data
            print(f"Tool: {tool_call.tool_name}")
            print(f"Arguments: {tool_call.arguments}")
        
        elif event.type == StreamingEventType.COMPLETE:
            print("Stream completed!")
            break
```

### StreamingChunk

Individual content chunks with progressive content:

```python
from jaf.core.streaming import StreamingChunk

# StreamingChunk contains:
# - content: Full accumulated content so far
# - delta: New content added in this chunk
# - is_complete: Whether this is the final chunk
# - token_count: Optional token count for the chunk

chunk = StreamingChunk(
    content="Hello, this is a streaming response",
    delta=" response",
    is_complete=False,
    token_count=5
)

print(f"Full content: {chunk.content}")
print(f"New content: {chunk.delta}")
print(f"Is final: {chunk.is_complete}")
```

## Advanced Features

### StreamingBuffer

Accumulate and manage streaming content:

```python
from jaf.core.streaming import StreamingBuffer, StreamingChunk

# Create buffer to accumulate streaming content
buffer = StreamingBuffer()

# Add chunks as they arrive
chunk1 = StreamingChunk(content="Hello", delta="Hello", is_complete=False)
chunk2 = StreamingChunk(content="Hello, world", delta=", world", is_complete=True)

buffer.add_chunk(chunk1)
buffer.add_chunk(chunk2)

# Get accumulated content
print(f"Full content: {buffer.content}")
print(f"Is complete: {buffer.is_complete}")

# Get final message
final_message = buffer.get_final_message()
print(f"Final message: {final_message.content}")
```

### StreamingCollector

Collect and replay streaming events:

```python
from jaf.core.streaming import StreamingCollector, run_streaming

# Create collector
collector = StreamingCollector()

# Collect stream events
async def collect_and_analyze():
    # Create stream
    stream = run_streaming(initial_state, config)
    
    # Collect all events
    buffer = await collector.collect_stream(stream, run_id="demo_run")
    
    # Analyze collected events
    events = collector.get_events_for_run("demo_run")
    print(f"Collected {len(events)} events")
    
    # Replay stream with delay
    async for event in collector.replay_stream("demo_run", delay_ms=100):
        print(f"Replaying: {event.type} - {event.timestamp}")
```

### Event Types and Data

Handle different types of streaming events:

```python
from jaf.core.streaming import StreamingEventType, StreamingEvent

async def handle_streaming_events(stream):
    async for event in stream:
        if event.type == StreamingEventType.START:
            metadata = event.data
            print(f"Stream started for agent: {metadata.agent_name}")
        
        elif event.type == StreamingEventType.CHUNK:
            chunk = event.data
            print(f"Content chunk: {chunk.delta}")
        
        elif event.type == StreamingEventType.TOOL_CALL:
            tool_call = event.data
            print(f"Tool called: {tool_call.tool_name}")
            print(f"Arguments: {tool_call.arguments}")
        
        elif event.type == StreamingEventType.TOOL_RESULT:
            tool_result = event.data
            print(f"Tool result: {tool_result.result}")
        
        elif event.type == StreamingEventType.ERROR:
            error = event.data
            print(f"Error occurred: {error}")
        
        elif event.type == StreamingEventType.COMPLETE:
            print("Stream completed successfully")
```

## Integration Examples

### FastAPI Integration

Integrate streaming with FastAPI:

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from jaf.core.streaming import StreamingAgent

app = FastAPI()

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    
    streaming_agent = StreamingAgent()
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            # Stream response
            async for chunk in streaming_agent.stream_response(message):
                await websocket.send_json({
                    'type': 'chunk',
                    'content': chunk.content,
                    'metadata': chunk.metadata
                })
            
            # Send completion signal
            await websocket.send_json({'type': 'complete'})
            
    except WebSocketDisconnect:
        logger.info("Client disconnected from streaming session")

@app.get("/stream-http")
async def http_stream(query: str):
    """HTTP streaming endpoint."""
    streaming_agent = StreamingAgent()
    
    async def generate_stream():
        async for chunk in streaming_agent.stream_response(query):
            yield f"data: {chunk.content}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )
```

### React Frontend Integration

Example React component for consuming streams:

```javascript
// StreamingChat.jsx
import React, { useState, useEffect } from 'react';

const StreamingChat = () => {
    const [messages, setMessages] = useState([]);
    const [currentResponse, setCurrentResponse] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);

    const sendMessage = async (message) => {
        setIsStreaming(true);
        setCurrentResponse('');
        
        try {
            const response = await fetch('/api/stream-http?query=' + encodeURIComponent(message));
            const reader = response.body.getReader();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = new TextDecoder().decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const content = line.slice(6);
                        setCurrentResponse(prev => prev + content);
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
        } finally {
            setIsStreaming(false);
            setMessages(prev => [...prev, { role: 'assistant', content: currentResponse }]);
            setCurrentResponse('');
        }
    };

    return (
        <div className="streaming-chat">
            <div className="messages">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.role}`}>
                        {msg.content}
                    </div>
                ))}
                {isStreaming && (
                    <div className="message assistant streaming">
                        {currentResponse}
                        <span className="cursor">|</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default StreamingChat;
```

## Best Practices

### 1. Optimize Chunk Sizes

Choose appropriate chunk sizes for your use case:

```python
# Good: Context-aware chunk sizing
def calculate_optimal_chunk_size(content_type: str, user_context: dict) -> int:
    base_sizes = {
        'code': 100,      # Larger chunks for code
        'explanation': 50, # Medium chunks for explanations
        'conversation': 30 # Smaller chunks for chat
    }
    
    base_size = base_sizes.get(content_type, 50)
    
    # Adjust for user preferences
    if user_context.get('reading_speed') == 'fast':
        return int(base_size * 1.5)
    elif user_context.get('reading_speed') == 'slow':
        return int(base_size * 0.7)
    
    return base_size
```

### 2. Implement Proper Error Boundaries

Handle errors without breaking the stream:

```python
# Good: Error boundary pattern
async def safe_streaming_generator(content_generator):
    try:
        async for chunk in content_generator:
            yield chunk
    except Exception as e:
        # Send error chunk instead of breaking stream
        error_chunk = StreamChunk(
            content=f"[Error: {str(e)}]",
            chunk_type=ChunkType.ERROR,
            is_final=True
        )
        yield error_chunk
```

### 3. Monitor Performance Continuously

Track key metrics for optimization:

```python
# Good: Comprehensive monitoring
class StreamingMetricsCollector:
    def __init__(self):
        self.metrics = {
            'total_streams': 0,
            'avg_chunk_size': 0,
            'avg_stream_duration': 0,
            'error_rate': 0,
            'user_engagement': 0
        }
    
    def track_stream_completion(self, duration_ms: int, chunk_count: int, user_stayed: bool):
        self.metrics['total_streams'] += 1
        self.metrics['avg_stream_duration'] = (
            (self.metrics['avg_stream_duration'] * (self.metrics['total_streams'] - 1) + duration_ms) 
            / self.metrics['total_streams']
        )
        
        if user_stayed:
            self.metrics['user_engagement'] += 1
```

## Example: Complete Streaming Implementation

Here's a comprehensive example showing a complete streaming implementation:

```python
import asyncio
import time
from typing import AsyncGenerator
from jaf.core.streaming import StreamingEngine, StreamConfig, StreamChunk
from jaf.core.analytics import AnalyticsEngine

class ComprehensiveStreamingAgent:
    """Complete streaming agent with analytics, error handling, and optimization."""
    
    def __init__(self):
        self.streaming_engine = StreamingEngine()
        self.analytics = AnalyticsEngine()
        self.active_streams = {}
    
    async def stream_response(
        self, 
        query: str, 
        user_context: dict,
        session_id: str
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a complete response with all features enabled."""
        
        # Start analytics tracking
        stream_session = self.analytics.start_streaming_session(
            session_id=session_id,
            query=query,
            user_context=user_context
        )
        
        try:
            # Generate response content
            full_response = await self._generate_response(query, user_context)
            
            # Configure streaming based on context
            config = self._create_stream_config(user_context)
            
            # Create optimized chunks
            chunks = self._create_optimized_chunks(full_response, config)
            
            # Stream with analytics and error handling
            chunk_count = 0
            start_time = time.time()
            
            for chunk in chunks:
                try:
                    # Apply flow control
                    await self._apply_flow_control(session_id, chunk)
                    
                    # Track chunk delivery
                    chunk_start = time.time()
                    yield chunk
                    chunk_end = time.time()
                    
                    # Record analytics
                    self.analytics.record_chunk_delivery(
                        session_id=session_id,
                        chunk_size=len(chunk.content),
                        delivery_time_ms=(chunk_end - chunk_start) * 1000
                    )
                    
                    chunk_count += 1
                    
                except Exception as e:
                    # Handle chunk-level errors
                    error_chunk = self._create_error_chunk(e, chunk_count)
                    yield error_chunk
                    break
            
            # Complete analytics
            total_time = time.time() - start_time
            self.analytics.complete_streaming_session(
                session_id=session_id,
                total_chunks=chunk_count,
                total_time_ms=total_time * 1000,
                success=True
            )
            
        except Exception as e:
            # Handle session-level errors
            self.analytics.record_streaming_error(session_id, str(e))
            error_chunk = self._create_error_chunk(e, 0)
            yield error_chunk
    
    def _create_stream_config(self, user_context: dict) -> StreamConfig:
        """Create optimized stream configuration."""
        return StreamConfig(
            chunk_size=self._calculate_chunk_size(user_context),
            delay_ms=self._calculate_delay(user_context),
            enable_sentence_boundaries=True,
            compression_enabled=user_context.get('low_bandwidth', False)
        )
    
    def _calculate_chunk_size(self, user_context: dict) -> int:
        """Calculate optimal chunk size based on user context."""
        base_size = 50
        
        # Adjust for device type
        if user_context.get('device_type') == 'mobile':
            base_size = 30
        elif user_context.get('device_type') == 'desktop':
            base_size = 70
        
        # Adjust for reading speed
        reading_speed = user_context.get('reading_speed', 'medium')
        if reading_speed == 'fast':
            base_size = int(base_size * 1.5)
        elif reading_speed == 'slow':
            base_size = int(base_size * 0.7)
        
        return max(20, min(base_size, 100))  # Clamp between 20-100
    
    async def _generate_response(self, query: str, user_context: dict) -> str:
        """Generate the complete response content."""
        # This would integrate with your LLM or agent system
        return f"This is a comprehensive response to: {query}. " * 10
    
    def _create_optimized_chunks(self, content: str, config: StreamConfig) -> list[StreamChunk]:
        """Create optimized chunks from content."""
        chunks = []
        chunk_size = config.chunk_size
        
        # Split content into chunks with sentence boundary awareness
        if config.enable_sentence_boundaries:
            sentences = content.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) <= chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(StreamChunk(
                            content=current_chunk.strip(),
                            chunk_id=f"chunk_{len(chunks)}",
                            sequence_number=len(chunks)
                        ))
                    current_chunk = sentence + ". "
            
            if current_chunk:
                chunks.append(StreamChunk(
                    content=current_chunk.strip(),
                    chunk_id=f"chunk_{len(chunks)}",
                    sequence_number=len(chunks),
                    is_final=True
                ))
        else:
            # Simple character-based chunking
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                chunks.append(StreamChunk(
                    content=chunk_content,
                    chunk_id=f"chunk_{len(chunks)}",
                    sequence_number=len(chunks),
                    is_final=(i + chunk_size >= len(content))
                ))
        
        return chunks

# Usage example
async def main():
    agent = ComprehensiveStreamingAgent()
    
    user_context = {
        'device_type': 'desktop',
        'reading_speed': 'medium',
        'low_bandwidth': False,
        'preferred_style': 'detailed'
    }
    
    print("🌊 Starting comprehensive streaming demo...")
    
    async for chunk in agent.stream_response(
        query="Explain machine learning concepts",
        user_context=user_context,
        session_id="demo_session_001"
    ):
        print(f"📝 Chunk {chunk.sequence_number}: {chunk.content}")
        await asyncio.sleep(0.1)  # Simulate processing time
    
    print("✅ Streaming completed!")

if __name__ == "__main__":
    asyncio.run(main())
```

The streaming response system provides a foundation for building engaging, real-time user experiences while maintaining performance and reliability.

## Next Steps

- Learn about [Analytics System](analytics-system.md) for streaming insights
- Explore [Workflow Orchestration](workflow-orchestration.md) for complex automation
- Check [Performance Monitoring](performance-monitoring.md) for optimization
- Review [Plugin System](plugin-system.md) for extensibility
