"""
Tracing and observability for the JAF framework.

This module provides tracing capabilities to monitor agent execution,
tool calls, and performance metrics.
"""
import os
os.environ["LANGFUSE_ENABLE_OTEL"] = "false"
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
import uuid

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from langfuse import Langfuse

from .types import TraceEvent, TraceId

# Global tracer provider
provider = None
tracer = None

def setup_otel_tracing(service_name: str = "jaf-agent", collector_url: Optional[str] = None) -> None:
    """Configure OpenTelemetry tracing."""
    global provider, tracer
    if not collector_url:
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    
    exporter = OTLPSpanExporter(endpoint=collector_url)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)


class OtelTraceCollector:
    """
    OpenTelemetry trace collector.

    NOTE: This implementation assumes a single active trace at a time per collector
    instance. It is suitable for simple scripts but not for concurrent environments
    like a web server handling multiple requests without proper scoping.
    """

    def __init__(self, service_name: str = "jaf-agent"):
        self.service_name = service_name
        self.tracer = trace.get_tracer(self.service_name)
        self.active_root_span: Optional[Any] = None

    def collect(self, event: TraceEvent) -> None:
        """Convert JAF event to OTEL span."""
        from dataclasses import asdict, is_dataclass

        if not self.tracer:
            return

        event_type = event.type
        data = event.data or {}

        if is_dataclass(data):
            data_dict = asdict(data)
        else:
            data_dict = data

        if event_type == "run_start":
            if self.active_root_span:
                self.active_root_span.end()

            trace_id = data_dict.get("trace_id") or data_dict.get("traceId")
            self.active_root_span = self.tracer.start_span(f"jaf.run.{trace_id}")
            ctx = trace.set_span_in_context(self.active_root_span)
        else:
            if not self.active_root_span:
                return
            ctx = trace.set_span_in_context(self.active_root_span)

        span_name = f"jaf.{event_type}"
        with self.tracer.start_as_current_span(span_name, context=ctx) as span:
            for key, value in data_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                elif value is not None:
                    try:
                        # Attempt to serialize complex objects to JSON
                        span.set_attribute(key, json.dumps(value, default=str))
                    except (TypeError, OverflowError):
                        # Fallback to string representation if serialization fails
                        span.set_attribute(key, str(value))

        if event_type == "run_end":
            if self.active_root_span:
                self.active_root_span.end()
                self.active_root_span = None

    def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
        """Not implemented for OTEL."""
        return []

    def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
        """Not implemented for OTEL."""
        return {}

    def clear(self, trace_id: Optional[TraceId] = None) -> None:
        """Not implemented for OTEL."""
        pass

class TraceCollector(Protocol):
    """Protocol for trace collectors."""

    def collect(self, event: TraceEvent) -> None:
        """Collect a trace event."""
        ...

    def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
        """Get all events for a specific trace."""
        ...

    def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
        """Get all traces."""
        ...

    def clear(self, trace_id: Optional[TraceId] = None) -> None:
        """Clear traces."""
        ...

class InMemoryTraceCollector:
    """In-memory trace collector that organizes events by trace ID."""

    def __init__(self):
        self.traces: Dict[TraceId, List[TraceEvent]] = {}

    def collect(self, event: TraceEvent) -> None:
        """Collect a trace event."""
        trace_id: Optional[TraceId] = None

        # Extract trace ID from event data - handle both snake_case and camelCase
        if hasattr(event, 'data') and isinstance(event.data, dict):
            # Try snake_case first (Python convention)
            if 'trace_id' in event.data:
                trace_id = event.data['trace_id']
            elif 'run_id' in event.data:
                trace_id = TraceId(event.data['run_id'])
            # Fallback to camelCase (for compatibility)
            elif 'traceId' in event.data:
                trace_id = event.data['traceId']
            elif 'runId' in event.data:
                trace_id = TraceId(event.data['runId'])

        if not trace_id:
            return

        if trace_id not in self.traces:
            self.traces[trace_id] = []

        self.traces[trace_id].append(event)

    def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
        """Get all events for a specific trace."""
        return self.traces.get(trace_id, [])

    def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
        """Get all traces."""
        return dict(self.traces)

    def clear(self, trace_id: Optional[TraceId] = None) -> None:
        """Clear traces."""
        if trace_id:
            self.traces.pop(trace_id, None)
        else:
            self.traces.clear()

class ConsoleTraceCollector:
    """Console trace collector with detailed logging."""

    def __init__(self):
        self.in_memory = InMemoryTraceCollector()
        self.run_start_times = {}  # Track start times per run

    def collect(self, event: TraceEvent) -> None:
        """Collect event and log to console."""
        self.in_memory.collect(event)

        timestamp = datetime.now().isoformat()
        prefix = f"[{timestamp}] JAF:{event.type}"

        if event.type == 'run_start':
            data = event.data
            run_id = data.get('run_id') or data.get('runId')
            trace_id = data.get('trace_id') or data.get('traceId')
            # Track start time for this run
            if run_id:
                self.run_start_times[run_id] = time.time()
            print(f"{prefix} Starting run {run_id} (trace: {trace_id})")

        elif event.type == 'llm_call_start':
            data = event.data
            model = data.get('model')
            agent_name = data.get('agent_name') or data.get('agentName')
            print(f"{prefix} Calling {model} for agent {agent_name}")

        elif event.type == 'llm_call_end':
            data = event.data
            choice = data.get('choice', {})
            message = choice.get('message', {}) if isinstance(choice, dict) else {}

            # Check for tool_calls with both naming conventions
            tool_calls = message.get('tool_calls') or message.get('toolCalls')
            has_tools = bool(tool_calls and len(tool_calls) > 0)
            has_content = bool(message.get('content'))

            if has_tools:
                response_type = 'tool calls'
            elif has_content:
                response_type = 'content'
            else:
                response_type = 'empty response'

            print(f"{prefix} LLM responded with {response_type}")

        elif event.type == 'tool_call_start':
            data = event.data
            tool_name = data.get('tool_name') or data.get('toolName')
            args = data.get('args')
            print(f"{prefix} Executing tool {tool_name} with args:", args)

        elif event.type == 'tool_call_end':
            data = event.data
            tool_name = data.get('tool_name') or data.get('toolName')
            print(f"{prefix} Tool {tool_name} completed")

        elif event.type == 'handoff':
            data = event.data
            from_agent = data.get('from')
            to_agent = data.get('to')
            print(f"{prefix} Agent handoff: {from_agent} → {to_agent}")

        elif event.type == 'run_end':
            data = event.data
            outcome = data.get('outcome')

            # Calculate elapsed time if we have a start time
            elapsed = None
            # Try to get run_id from outcome or use a fallback
            run_id = None
            if outcome and hasattr(outcome, 'final_state') and hasattr(outcome.final_state, 'run_id'):
                run_id = outcome.final_state.run_id

            if run_id and run_id in self.run_start_times:
                elapsed = time.time() - self.run_start_times[run_id]
                # Clean up the start time
                del self.run_start_times[run_id]

            if outcome and hasattr(outcome, 'status'):
                status = outcome.status

                if status == 'completed':
                    elapsed_str = f" in {elapsed:.2f}s" if elapsed else ""
                    print(f"{prefix} Run completed successfully{elapsed_str}")
                else:
                    error = outcome.error if hasattr(outcome, 'error') else None
                    error_type = getattr(error, '_tag', 'unknown') if error and hasattr(error, '_tag') else 'unknown'
                    elapsed_str = f" in {elapsed:.2f}s" if elapsed else ""
                    print(f"{prefix} Run failed with {error_type}{elapsed_str}")
            else:
                elapsed_str = f" in {elapsed:.2f}s" if elapsed else ""
                print(f"{prefix} Run completed{elapsed_str}")

    def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
        """Get all events for a specific trace."""
        return self.in_memory.get_trace(trace_id)

    def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
        """Get all traces."""
        return self.in_memory.get_all_traces()

    def clear(self, trace_id: Optional[TraceId] = None) -> None:
        """Clear traces."""
        self.in_memory.clear(trace_id)

class FileTraceCollector:
    """File trace collector that writes events to a file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.in_memory = InMemoryTraceCollector()

    def collect(self, event: TraceEvent) -> None:
        """Collect event and write to file."""
        self.in_memory.collect(event)

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': event.type,
            'data': event.data
        }

        try:
            # Ensure directory exists if file path has a directory
            dir_path = os.path.dirname(self.file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
        except Exception as error:
            print(f"Failed to write trace to file: {error}")

    def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
        """Get all events for a specific trace."""
        return self.in_memory.get_trace(trace_id)

    def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
        """Get all traces."""
        return self.in_memory.get_all_traces()

    def clear(self, trace_id: Optional[TraceId] = None) -> None:
        """Clear traces."""
        self.in_memory.clear(trace_id)

class LangfuseTraceCollector:
    """Langfuse trace collector using v2 SDK."""

    def __init__(self):
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST")
        
        print(f"[LANGFUSE] Initializing with host: {host}")
        print(f"[LANGFUSE] Public key: {public_key[:10]}..." if public_key else "[LANGFUSE] No public key set")
        print(f"[LANGFUSE] Secret key: {secret_key[:10]}..." if secret_key else "[LANGFUSE] No secret key set")
        
        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            release="jaf-py-v2.4.8"
        )
        self.active_spans: Dict[str, Any] = {}
        self.trace_spans: Dict[TraceId, Any] = {}
        # Track tool calls and results for each trace
        self.trace_tool_calls: Dict[TraceId, List[Dict[str, Any]]] = {}
        self.trace_tool_results: Dict[TraceId, List[Dict[str, Any]]] = {}

    def collect(self, event: TraceEvent) -> None:
        """Collect a trace event and send it to Langfuse."""
        try:
            trace_id = self._get_trace_id(event)
            if not trace_id:
                print(f"[LANGFUSE] No trace_id found for event: {event.type}")
                return

            print(f"[LANGFUSE] Processing event: {event.type} for trace: {trace_id}")

            if event.type == "run_start":
                # Start a new trace for the entire run
                print(f"[LANGFUSE] Starting trace for run: {trace_id}")
                
                # Initialize tracking for this trace
                self.trace_tool_calls[trace_id] = []
                self.trace_tool_results[trace_id] = []
                
                # Extract user query from the run_start data
                user_query = None
                user_id = None
                conversation_history = []
                
                # Debug: Print the event data structure to understand what we're working with
                if event.data.get("context"):
                    context = event.data["context"]
                    print(f"[LANGFUSE DEBUG] Context type: {type(context)}")
                    print(f"[LANGFUSE DEBUG] Context attributes: {dir(context) if hasattr(context, '__dict__') else 'Not an object'}")
                    if hasattr(context, '__dict__'):
                        print(f"[LANGFUSE DEBUG] Context dict: {context.__dict__}")
                
                # Try to extract from context first
                context = event.data.get("context")
                if context:
                    # Try direct attribute access
                    if hasattr(context, 'query'):
                        user_query = context.query
                        print(f"[LANGFUSE DEBUG] Found user_query from context.query: {user_query}")
                    
                    # Try to extract from combined_history
                    if hasattr(context, 'combined_history') and context.combined_history:
                        history = context.combined_history
                        print(f"[LANGFUSE DEBUG] Found combined_history with {len(history)} messages")
                        for i, msg in enumerate(reversed(history)):
                            print(f"[LANGFUSE DEBUG] History message {i}: {msg}")
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                user_query = msg.get("content", "")
                                print(f"[LANGFUSE DEBUG] Found user_query from history: {user_query}")
                                break
                    
                    # Try to extract user_id from user_info
                    if hasattr(context, 'user_info'):
                        user_info = context.user_info
                        print(f"[LANGFUSE DEBUG] Found user_info: {type(user_info)}")
                        if isinstance(user_info, dict):
                            user_id = user_info.get("email") or user_info.get("username")
                            print(f"[LANGFUSE DEBUG] Extracted user_id: {user_id}")
                        elif hasattr(user_info, 'email'):
                            user_id = user_info.email
                            print(f"[LANGFUSE DEBUG] Extracted user_id from attr: {user_id}")
                
                # Extract conversation history and current user query from messages
                messages = event.data.get("messages", [])
                if messages:
                    print(f"[LANGFUSE DEBUG] Processing {len(messages)} messages")
                    
                    # Find the last user message (current query) and extract conversation history (excluding current)
                    current_user_message_found = False
                    for i in range(len(messages) - 1, -1, -1):
                        msg = messages[i]
                        
                        # Extract message data comprehensively
                        msg_data = {}
                        
                        if isinstance(msg, dict):
                            role = msg.get("role")
                            content = msg.get("content", "")
                            # Capture all additional fields from dict messages
                            msg_data = {
                                "role": role,
                                "content": content,
                                "tool_calls": msg.get("tool_calls"),
                                "tool_call_id": msg.get("tool_call_id"),
                                "name": msg.get("name"),
                                "function_call": msg.get("function_call"),
                                "timestamp": msg.get("timestamp", datetime.now().isoformat())
                            }
                        elif hasattr(msg, 'role'):
                            role = getattr(msg, 'role', None)
                            content = getattr(msg, 'content', "")
                            # Handle both string content and complex content structures
                            if not isinstance(content, str):
                                # Try to extract text from complex content
                                if hasattr(content, '__iter__') and not isinstance(content, str):
                                    try:
                                        # If it's a list, try to join text parts
                                        content = " ".join(str(item) for item in content if item)
                                    except:
                                        content = str(content)
                                else:
                                    content = str(content)
                            
                            # Capture all additional fields from object messages
                            msg_data = {
                                "role": role,
                                "content": content,
                                "tool_calls": getattr(msg, 'tool_calls', None),
                                "tool_call_id": getattr(msg, 'tool_call_id', None),
                                "name": getattr(msg, 'name', None),
                                "function_call": getattr(msg, 'function_call', None),
                                "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                            }
                        else:
                            # Handle messages that don't have expected structure
                            print(f"[LANGFUSE DEBUG] Skipping message with unexpected structure: {type(msg)}")
                            continue
                        
                        # Clean up None values from msg_data
                        msg_data = {k: v for k, v in msg_data.items() if v is not None}
                        
                        # If we haven't found the current user message yet and this is a user message
                        if not current_user_message_found and (role == "user" or role == 'user'):
                            user_query = content
                            current_user_message_found = True
                            print(f"[LANGFUSE DEBUG] Found current user query: {user_query}")
                        elif current_user_message_found:
                            # Add to conversation history (excluding the current user message)
                            # Include ALL message types: assistant, tool, system, function, etc.
                            conversation_history.insert(0, msg_data)
                            print(f"[LANGFUSE DEBUG] Added to conversation history: role={role}, content_length={len(str(content))}, has_tool_calls={bool(msg_data.get('tool_calls'))}")
                
                print(f"[LANGFUSE DEBUG] Final extracted - user_query: {user_query}, user_id: {user_id}")
                print(f"[LANGFUSE DEBUG] Conversation history length: {len(conversation_history)}")
                
                # Debug: Log the roles and types captured in conversation history
                if conversation_history:
                    roles_summary = {}
                    for msg in conversation_history:
                        role = msg.get("role", "unknown")
                        roles_summary[role] = roles_summary.get(role, 0) + 1
                    print(f"[LANGFUSE DEBUG] Conversation history roles breakdown: {roles_summary}")
                    
                    # Log first few messages for verification
                    for i, msg in enumerate(conversation_history[:3]):
                        role = msg.get("role", "unknown")
                        content_preview = str(msg.get("content", ""))[:100]
                        has_tool_calls = bool(msg.get("tool_calls"))
                        has_tool_call_id = bool(msg.get("tool_call_id"))
                        print(f"[LANGFUSE DEBUG] History msg {i}: role={role}, content='{content_preview}...', tool_calls={has_tool_calls}, tool_call_id={has_tool_call_id}")
                
                # Create comprehensive input data for the trace
                trace_input = {
                    "user_query": user_query,
                    "run_id": str(trace_id),
                    "agent_name": event.data.get("agent_name", "analytics_agent_jaf"),
                    "session_info": {
                        "session_id": event.data.get("session_id"),
                        "user_id": user_id or event.data.get("user_id")
                    }
                }
                
                trace = self.langfuse.trace(
                    name=f"jaf-run-{trace_id}",
                    user_id=user_id or event.data.get("user_id"),
                    session_id=event.data.get("session_id"),
                    input=trace_input,
                    metadata={
                        "framework": "jaf",
                        "event_type": "run_start",
                        "trace_id": str(trace_id),
                        "user_query": user_query,
                        "user_id": user_id or event.data.get("user_id"),
                        "agent_name": event.data.get("agent_name", "analytics_agent_jaf"),
                        "conversation_history": conversation_history,
                        "tool_calls": [],
                        "tool_results": [],
                        "user_info": event.data.get("context").user_info if event.data.get("context") and hasattr(event.data.get("context"), 'user_info') else None
                    }
                )
                self.trace_spans[trace_id] = trace
                # Store user_id, user_query, and conversation_history for later use
                trace._user_id = user_id or event.data.get("user_id")
                trace._user_query = user_query
                trace._conversation_history = conversation_history
                print(f"[LANGFUSE] Created trace with user query: {user_query[:100] if user_query else 'None'}...")
                
            elif event.type == "run_end":
                if trace_id in self.trace_spans:
                    print(f"[LANGFUSE] Ending trace for run: {trace_id}")
                    
                    # Update the trace metadata with final tool calls and results
                    conversation_history = getattr(self.trace_spans[trace_id], '_conversation_history', [])
                    final_metadata = {
                        "framework": "jaf",
                        "event_type": "run_end",
                        "trace_id": str(trace_id),
                        "user_query": getattr(self.trace_spans[trace_id], '_user_query', None),
                        "user_id": getattr(self.trace_spans[trace_id], '_user_id', None),
                        "agent_name": event.data.get("agent_name", "analytics_agent_jaf"),
                        "conversation_history": conversation_history,
                        "tool_calls": self.trace_tool_calls.get(trace_id, []),
                        "tool_results": self.trace_tool_results.get(trace_id, [])
                    }
                    
                    # End the trace with updated metadata
                    self.trace_spans[trace_id].update(
                        output=event.data,
                        metadata=final_metadata
                    )
                    
                    # Flush to ensure data is sent
                    print(f"[LANGFUSE] Flushing data to Langfuse...")
                    self.langfuse.flush()
                    print(f"[LANGFUSE] Flush completed")
                    
                    # Clean up
                    del self.trace_spans[trace_id]
                    if trace_id in self.trace_tool_calls:
                        del self.trace_tool_calls[trace_id]
                    if trace_id in self.trace_tool_results:
                        del self.trace_tool_results[trace_id]
                else:
                    print(f"[LANGFUSE] No trace found for run_end: {trace_id}")
                    
            elif event.type == "llm_call_start":
                # Start a generation for LLM calls
                model = event.data.get("model", "unknown")
                print(f"[LANGFUSE] Starting generation for LLM call with model: {model}")
                
                # Get stored user information from the trace
                trace = self.trace_spans[trace_id]
                user_id = getattr(trace, '_user_id', None)
                user_query = getattr(trace, '_user_query', None)
                
                generation = trace.generation(
                    name=f"llm-call-{model}",
                    input=event.data.get("messages"),
                    metadata={
                        "agent_name": event.data.get("agent_name"), 
                        "model": model,
                        "user_id": user_id,
                        "user_query": user_query
                    }
                )
                span_id = self._get_span_id(event)
                self.active_spans[span_id] = generation
                print(f"[LANGFUSE] Created generation: {generation}")
                    
            elif event.type == "llm_call_end":
                span_id = self._get_span_id(event)
                if span_id in self.active_spans:
                    print(f"[LANGFUSE] Ending generation for LLM call")
                    # End the generation
                    generation = self.active_spans[span_id]
                    choice = event.data.get("choice", {})
                    
                    # Extract usage from the event data
                    usage = event.data.get("usage", {})
                    
                    # Extract model information from choice data or event data
                    model = choice.get("model", "unknown")
                    if model == "unknown":
                        # Try to get model from the choice response structure
                        if isinstance(choice, dict):
                            model = choice.get("model") or choice.get("id", "unknown")
                    
                    # Convert to Langfuse v2 format - let Langfuse handle cost calculation automatically
                    langfuse_usage = None
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        
                        langfuse_usage = {
                            "input": prompt_tokens,
                            "output": completion_tokens,
                            "total": total_tokens,
                            "unit": "TOKENS"
                        }
                        
                        print(f"[LANGFUSE] Usage data for automatic cost calculation: {langfuse_usage}")

                    # Include model information in the generation end - Langfuse will calculate costs automatically
                    generation.end(
                        output=choice, 
                        usage=langfuse_usage,
                        model=model,  # Pass model directly for automatic cost calculation
                        metadata={
                            "model": model,
                            "system_fingerprint": choice.get("system_fingerprint"),
                            "created": choice.get("created"),
                            "response_id": choice.get("id")
                        }
                    )

                    # Clean up the span reference
                    del self.active_spans[span_id]
                    print(f"[LANGFUSE] Generation ended with cost tracking")
                else:
                    print(f"[LANGFUSE] No generation found for llm_call_end: {span_id}")
                    
            elif event.type == "tool_call_start":
                # Start a span for tool calls with detailed input information
                tool_name = event.data.get('tool_name', 'unknown')
                tool_args = event.data.get("args", {})
                call_id = event.data.get("call_id")
                if not call_id:
                    call_id = f"{tool_name}-{uuid.uuid4().hex[:8]}"
                    try:
                        event.data["call_id"] = call_id
                    except TypeError:
                        # event.data may be immutable; log and rely on synthetic ID tracking downstream
                        print(f"[LANGFUSE] Generated synthetic call_id for tool start: {call_id}")
                
                print(f"[LANGFUSE] Starting span for tool call: {tool_name} ({call_id})")
                
                # Track this tool call for the trace
                tool_call_data = {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "call_id": call_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Ensure trace_id exists in tracking
                if trace_id not in self.trace_tool_calls:
                    self.trace_tool_calls[trace_id] = []

                self.trace_tool_calls[trace_id].append(tool_call_data)
                
                # Create comprehensive input data for the tool call
                tool_input = {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "call_id": call_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                span = self.trace_spans[trace_id].span(
                    name=f"tool-{tool_name}",
                    input=tool_input,
                    metadata={
                        "tool_name": tool_name,
                        "call_id": call_id,
                        "framework": "jaf",
                        "event_type": "tool_call"
                    }
                )
                span_id = self._get_span_id(event)
                self.active_spans[span_id] = span
                print(f"[LANGFUSE] Created tool span for {tool_name} with args: {str(tool_args)[:100]}...")
                    
            elif event.type == "tool_call_end":
                span_id = self._get_span_id(event)
                if span_id in self.active_spans:
                    tool_name = event.data.get('tool_name', 'unknown')
                    tool_result = event.data.get("result")
                    call_id = event.data.get("call_id")
                    
                    print(f"[LANGFUSE] Ending span for tool call: {tool_name} ({call_id})")
                    
                    # Track this tool result for the trace
                    tool_result_data = {
                        "tool_name": tool_name,
                        "result": tool_result,
                        "call_id": call_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": event.data.get("status", "completed"),
                        "tool_result": event.data.get("tool_result")
                    }
                    
                    if trace_id not in self.trace_tool_results:
                        self.trace_tool_results[trace_id] = []
                    
                    self.trace_tool_results[trace_id].append(tool_result_data)
                    
                    # Create comprehensive output data for the tool call
                    tool_output = {
                        "tool_name": tool_name,
                        "result": tool_result,
                        "call_id": call_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": event.data.get("status", "completed")
                    }
                    
                    # End the span with detailed output
                    span = self.active_spans[span_id]
                    span.end(
                        output=tool_output,
                        metadata={
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "result_length": len(str(tool_result)) if tool_result else 0,
                            "framework": "jaf",
                            "event_type": "tool_call_end"
                        }
                    )
                    
                    # Clean up the span reference
                    del self.active_spans[span_id]
                    print(f"[LANGFUSE] Tool span ended for {tool_name} with result length: {len(str(tool_result)) if tool_result else 0}")
                else:
                    print(f"[LANGFUSE] No tool span found for tool_call_end: {span_id}")
                    
            elif event.type == "handoff":
                # Create an event for handoffs
                print(f"[LANGFUSE] Creating event for handoff")
                self.trace_spans[trace_id].event(
                    name="agent-handoff",
                    input={"from": event.data.get("from"), "to": event.data.get("to")},
                    metadata=event.data
                )
                print(f"[LANGFUSE] Handoff event created")
                    
            else:
                # Create a generic event for other event types
                print(f"[LANGFUSE] Creating generic event for: {event.type}")
                self.trace_spans[trace_id].event(
                    name=event.type,
                    input=event.data,
                    metadata={"framework": "jaf", "event_type": event.type}
                )
                print(f"[LANGFUSE] Generic event created")
                    
        except Exception as e:
            # Log error but don't break the application
            print(f"[LANGFUSE] ERROR: Trace collection failed: {e}")
            import traceback
            traceback.print_exc()

    def _get_trace_id(self, event: TraceEvent) -> Optional[TraceId]:
        """Extract trace ID from event data."""
        if hasattr(event, 'data') and isinstance(event.data, dict):
            # Try snake_case first (Python convention)
            if 'trace_id' in event.data:
                return event.data['trace_id']
            elif 'run_id' in event.data:
                return TraceId(event.data['run_id'])
            # Fallback to camelCase (for compatibility)
            elif 'traceId' in event.data:
                return event.data['traceId']
            elif 'runId' in event.data:
                return TraceId(event.data['runId'])
        
        # Debug: print what's actually in the event data
        return None

    def _get_span_id(self, event: TraceEvent) -> str:
        """Generate a unique span ID for the event."""
        trace_id = self._get_trace_id(event)
        
        # Use consistent identifiers that don't depend on timestamp
        if event.type.startswith('tool_call'):
            call_id = event.data.get('call_id') or event.data.get('tool_call_id')
            if call_id:
                return f"tool-{trace_id}-{call_id}"
            tool_name = event.data.get('tool_name') or event.data.get('toolName', 'unknown')
            return f"tool-{tool_name}-{trace_id}"
        elif event.type.startswith('llm_call'):
            # For LLM calls, use a simpler consistent ID that matches between start and end
            # Get run_id for more consistent matching
            run_id = event.data.get('run_id') or event.data.get('runId', trace_id)
            return f"llm-{run_id}"
        else:
            return f"{event.type}-{trace_id}"

    def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
        """Not implemented for Langfuse."""
        return []

    def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
        """Not implemented for Langfuse."""
        return {}

    def clear(self, trace_id: Optional[TraceId] = None) -> None:
        """Not implemented for Langfuse."""
        pass


def create_composite_trace_collector(*collectors: TraceCollector) -> TraceCollector:
    """Create a composite trace collector that forwards events to multiple collectors."""
    
    collector_list = list(collectors)
    
    # Automatically add OTEL collector if URL is configured
    collector_url = os.getenv("TRACE_COLLECTOR_URL")
    if collector_url:
        setup_otel_tracing(collector_url=collector_url)
        otel_collector = OtelTraceCollector()
        collector_list.append(otel_collector)

    # Automatically add Langfuse collector if keys are configured
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse_collector = LangfuseTraceCollector()
        collector_list.append(langfuse_collector)

    class CompositeTraceCollector:
        def __init__(self, collectors_list: List[TraceCollector]):
            self.collectors = collectors_list

        def collect(self, event: TraceEvent) -> None:
            """Forward event to all collectors."""
            for collector in self.collectors:
                try:
                    collector.collect(event)
                except Exception as e:
                    print(f"Warning: Trace collector failed: {e}")

        def get_trace(self, trace_id: TraceId) -> List[TraceEvent]:
            """Get trace from first collector."""
            if self.collectors:
                return self.collectors[0].get_trace(trace_id)
            return []

        def get_all_traces(self) -> Dict[TraceId, List[TraceEvent]]:
            """Get all traces from first collector."""
            if self.collectors:
                return self.collectors[0].get_all_traces()
            return {}

        def clear(self, trace_id: Optional[TraceId] = None) -> None:
            """Clear traces in all collectors."""
            for collector in self.collectors:
                try:
                    collector.clear(trace_id)
                except Exception as e:
                    print(f"Warning: Failed to clear trace collector: {e}")

    return CompositeTraceCollector(collector_list)
