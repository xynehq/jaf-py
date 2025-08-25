"""
Streaming support for JAF framework.

This module provides real-time streaming capabilities for agent responses,
enabling progressive output delivery and improved user experience.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Any, Union, Callable
from enum import Enum

from .types import (
    RunState, RunConfig, Message, TraceEvent, RunId, TraceId,
    ContentRole, ToolCall, JAFError, CompletedOutcome, ErrorOutcome, ModelBehaviorError
)


class StreamingEventType(str, Enum):
    """Types of streaming events."""
    START = 'start'
    CHUNK = 'chunk'
    TOOL_CALL = 'tool_call'
    TOOL_RESULT = 'tool_result'
    AGENT_SWITCH = 'agent_switch'
    ERROR = 'error'
    COMPLETE = 'complete'
    METADATA = 'metadata'


@dataclass(frozen=True)
class StreamingChunk:
    """A chunk of streaming content."""
    content: str
    delta: str  # The new content added in this chunk
    is_complete: bool = False
    token_count: Optional[int] = None


@dataclass(frozen=True)
class StreamingToolCall:
    """Streaming tool call information."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str
    status: str = 'started'  # 'started', 'executing', 'completed', 'failed'


@dataclass(frozen=True)
class StreamingToolResult:
    """Streaming tool result information."""
    tool_name: str
    call_id: str
    result: Any
    status: str
    execution_time_ms: Optional[float] = None


@dataclass(frozen=True)
class StreamingMetadata:
    """Metadata about the streaming session."""
    agent_name: str
    model_name: str
    turn_count: int
    total_tokens: int
    execution_time_ms: float
    performance_metrics: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StreamingEvent:
    """A streaming event containing progressive updates."""
    type: StreamingEventType
    data: Union[StreamingChunk, StreamingToolCall, StreamingToolResult, StreamingMetadata, JAFError]
    timestamp: float = field(default_factory=time.time)
    run_id: Optional[RunId] = None
    trace_id: Optional[TraceId] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert streaming event to dictionary for serialization."""
        return {
            'type': self.type.value,
            'data': self._serialize_data(),
            'timestamp': self.timestamp,
            'run_id': str(self.run_id) if self.run_id else None,
            'trace_id': str(self.trace_id) if self.trace_id else None
        }
    
    def _serialize_data(self) -> Dict[str, Any]:
        """Serialize the data field based on its type."""
        if isinstance(self.data, (StreamingChunk, StreamingToolCall, StreamingToolResult, StreamingMetadata)):
            # Convert dataclass to dict
            result = {}
            for field_name, field_value in self.data.__dict__.items():
                if field_value is not None:
                    result[field_name] = field_value
            return result
        elif isinstance(self.data, JAFError):
            return {
                'error_type': self.data._tag,
                'detail': getattr(self.data, 'detail', str(self.data))
            }
        else:
            return {'value': self.data}
    
    def to_json(self) -> str:
        """Convert streaming event to JSON string."""
        return json.dumps(self.to_dict())


class StreamingBuffer:
    """
    Buffer for accumulating streaming content and managing state.
    """
    
    def __init__(self):
        self.content: str = ""
        self.chunks: List[StreamingChunk] = []
        self.tool_calls: List[StreamingToolCall] = []
        self.tool_results: List[StreamingToolResult] = []
        self.metadata: Optional[StreamingMetadata] = None
        self.is_complete: bool = False
        self.error: Optional[JAFError] = None
    
    def add_chunk(self, chunk: StreamingChunk) -> None:
        """Add a content chunk to the buffer."""
        self.chunks.append(chunk)
        self.content += chunk.delta
        if chunk.is_complete:
            self.is_complete = True
    
    def add_tool_call(self, tool_call: StreamingToolCall) -> None:
        """Add a tool call to the buffer."""
        self.tool_calls.append(tool_call)
    
    def add_tool_result(self, tool_result: StreamingToolResult) -> None:
        """Add a tool result to the buffer."""
        self.tool_results.append(tool_result)
    
    def set_metadata(self, metadata: StreamingMetadata) -> None:
        """Set session metadata."""
        self.metadata = metadata
    
    def set_error(self, error: JAFError) -> None:
        """Set error state."""
        self.error = error
        self.is_complete = True
    
    def get_final_message(self) -> Message:
        """Get the final accumulated message."""
        tool_calls = None
        if self.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.call_id,
                    type='function',
                    function={'name': tc.tool_name, 'arguments': json.dumps(tc.arguments)}
                )
                for tc in self.tool_calls
            ]
        
        return Message(
            role=ContentRole.ASSISTANT,
            content=self.content,
            tool_calls=tool_calls
        )


async def run_streaming(
    initial_state: RunState,
    config: RunConfig,
    chunk_size: int = 50,
    include_metadata: bool = True
) -> AsyncIterator[StreamingEvent]:
    """
    Run an agent with streaming output.
    
    This function provides real-time streaming of agent responses, tool calls,
    and execution metadata. It yields StreamingEvent objects that can be
    consumed by clients for progressive UI updates.
    
    Args:
        initial_state: Initial run state
        config: Run configuration
        chunk_size: Size of content chunks for streaming (characters)
        include_metadata: Whether to include performance metadata
        
    Yields:
        StreamingEvent: Progressive updates during execution
    """
    start_time = time.time()
    event_queue = asyncio.Queue()
    event_available = asyncio.Event()

    # Emit start event
    yield StreamingEvent(
        type=StreamingEventType.START,
        data=StreamingMetadata(
            agent_name=initial_state.current_agent_name,
            model_name="unknown",
            turn_count=initial_state.turn_count,
            total_tokens=0,
            execution_time_ms=0
        ),
        run_id=initial_state.run_id,
        trace_id=initial_state.trace_id
    )

    tool_call_ids = {} # To map tool calls to their IDs

    def event_handler(event: TraceEvent) -> None:
        """Handle trace events and put them into the queue."""
        nonlocal tool_call_ids
        streaming_event = None
        if event.type == 'tool_call_start':
            # Generate a unique ID for the tool call
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_call_ids[event.data.tool_name] = call_id
            
            tool_call = StreamingToolCall(
                tool_name=event.data.tool_name,
                arguments=event.data.args,
                call_id=call_id,
                status='started'
            )
            streaming_event = StreamingEvent(
                type=StreamingEventType.TOOL_CALL,
                data=tool_call,
                run_id=initial_state.run_id,
                trace_id=initial_state.trace_id
            )
        elif event.type == 'tool_call_end':
            if event.data.tool_name not in tool_call_ids:
                raise RuntimeError(
                    f"Tool call end event received for unknown tool '{event.data.tool_name}'. "
                    f"Known tool calls: {list(tool_call_ids.keys())}. "
                    f"This may indicate a missing tool_call_start event or a bug in the streaming implementation."
                )
            call_id = tool_call_ids[event.data.tool_name]
            tool_result = StreamingToolResult(
                tool_name=event.data.tool_name,
                call_id=call_id,
                result=event.data.result,
                status=event.data.status or 'completed'
            )
            streaming_event = StreamingEvent(
                type=StreamingEventType.TOOL_RESULT,
                data=tool_result,
                run_id=initial_state.run_id,
                trace_id=initial_state.trace_id
            )
        
        if streaming_event:
            try:
                event_queue.put_nowait(streaming_event)
                event_available.set()
            except asyncio.QueueFull:
                print(f"JAF-WARNING: Streaming event queue is full. Event dropped: {streaming_event.type}")

    streaming_config = RunConfig(
        agent_registry=config.agent_registry,
        model_provider=config.model_provider,
        max_turns=config.max_turns,
        model_override=config.model_override,
        initial_input_guardrails=config.initial_input_guardrails,
        final_output_guardrails=config.final_output_guardrails,
        on_event=event_handler,
        memory=config.memory,
        conversation_id=config.conversation_id
    )

    from .engine import run

    run_task = asyncio.create_task(run(initial_state, streaming_config))

    while not run_task.done() or not event_queue.empty():
        if event_queue.empty():
            await event_available.wait()
            event_available.clear()
            continue
        try:
            event = event_queue.get_nowait()
            yield event
            if event.type == StreamingEventType.ERROR:
                if not run_task.done():
                    run_task.cancel()
                return
        except asyncio.QueueEmpty:
            continue
        except asyncio.CancelledError:
            if not run_task.done():
                run_task.cancel()
            raise

    try:
        result = await run_task
    except Exception as e:
        error = ModelBehaviorError(detail=str(e))
        yield StreamingEvent(
            type=StreamingEventType.ERROR,
            data=error,
            run_id=initial_state.run_id,
            trace_id=initial_state.trace_id
        )
        return

    if result.outcome.status == 'completed':
        final_content = str(result.outcome.output) if result.outcome.output else ""
        
        # Stream content in chunks
        for i in range(0, len(final_content), chunk_size):
            chunk_content = final_content[i:i + chunk_size]
            is_final_chunk = i + chunk_size >= len(final_content)
            
            chunk = StreamingChunk(
                content=final_content[:i + len(chunk_content)],
                delta=chunk_content,
                is_complete=is_final_chunk,
                token_count=len(chunk_content.split()) if is_final_chunk else None
            )
            
            yield StreamingEvent(
                type=StreamingEventType.CHUNK,
                data=chunk,
                run_id=initial_state.run_id,
                trace_id=initial_state.trace_id
            )
            await asyncio.sleep(0.01) # simulate network latency

        execution_time = (time.time() - start_time) * 1000
        if include_metadata:
            metadata = StreamingMetadata(
                agent_name=initial_state.current_agent_name,
                model_name=config.model_override or "default",
                turn_count=result.final_state.turn_count,
                total_tokens=len(final_content.split()),
                execution_time_ms=execution_time
            )
            yield StreamingEvent(
                type=StreamingEventType.METADATA,
                data=metadata,
                run_id=initial_state.run_id,
                trace_id=initial_state.trace_id
            )
        
        yield StreamingEvent(
            type=StreamingEventType.COMPLETE,
            data=StreamingChunk(content=final_content, delta="", is_complete=True),
            run_id=initial_state.run_id,
            trace_id=initial_state.trace_id
        )
    else:
        yield StreamingEvent(
            type=StreamingEventType.ERROR,
            data=result.outcome.error,
            run_id=initial_state.run_id,
            trace_id=initial_state.trace_id
        )


class StreamingCollector:
    """
    Collects streaming events for analysis and replay.
    """
    
    def __init__(self):
        self.events: List[StreamingEvent] = []
        self.buffers: Dict[str, StreamingBuffer] = {}
    
    async def collect_stream(
        self,
        stream: AsyncIterator[StreamingEvent],
        run_id: Optional[str] = None
    ) -> StreamingBuffer:
        """
        Collect all events from a stream and return the final buffer.
        
        Args:
            stream: Async iterator of streaming events
            run_id: Optional run ID for tracking
            
        Returns:
            StreamingBuffer: Final accumulated buffer
        """
        buffer_key = run_id or "default"
        buffer = StreamingBuffer()
        self.buffers[buffer_key] = buffer
        
        async for event in stream:
            self.events.append(event)
            
            if event.type == StreamingEventType.CHUNK:
                buffer.add_chunk(event.data)
            elif event.type == StreamingEventType.TOOL_CALL:
                buffer.add_tool_call(event.data)
            elif event.type == StreamingEventType.TOOL_RESULT:
                buffer.add_tool_result(event.data)
            elif event.type == StreamingEventType.METADATA:
                buffer.set_metadata(event.data)
            elif event.type == StreamingEventType.ERROR:
                buffer.set_error(event.data)
            elif event.type == StreamingEventType.COMPLETE:
                buffer.is_complete = True
        
        return buffer
    
    def get_events_for_run(self, run_id: str) -> List[StreamingEvent]:
        """Get all events for a specific run."""
        return [
            event for event in self.events
            if event.run_id and str(event.run_id) == run_id
        ]
    
    def replay_stream(self, run_id: str, delay_ms: int = 50) -> AsyncIterator[StreamingEvent]:
        """
        Replay a collected stream with optional delay.
        
        Args:
            run_id: Run ID to replay
            delay_ms: Delay between events in milliseconds
            
        Yields:
            StreamingEvent: Replayed events
        """
        async def _replay():
            events = self.get_events_for_run(run_id)
            for event in events:
                yield event
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)
        
        return _replay()


# Utility functions for streaming integration

def create_sse_response(event: StreamingEvent) -> str:
    """
    Create a Server-Sent Events (SSE) formatted response.
    
    Args:
        event: Streaming event to format
        
    Returns:
        str: SSE-formatted string
    """
    return f"data: {event.to_json()}\n\n"


async def stream_to_websocket(
    stream: AsyncIterator[StreamingEvent],
    websocket_send: Callable[[str], None]
) -> None:
    """
    Stream events to a WebSocket connection.
    
    Args:
        stream: Stream of events
        websocket_send: WebSocket send function
    """
    async for event in stream:
        await websocket_send(event.to_json())


def create_streaming_middleware(
    on_event: Optional[Callable[[StreamingEvent], None]] = None
) -> Callable[[AsyncIterator[StreamingEvent]], AsyncIterator[StreamingEvent]]:
    """
    Create middleware for processing streaming events.
    
    Args:
        on_event: Optional callback for each event
        
    Returns:
        Middleware function
    """
    async def middleware(stream: AsyncIterator[StreamingEvent]) -> AsyncIterator[StreamingEvent]:
        async for event in stream:
            if on_event:
                on_event(event)
            yield event
    
    return middleware
