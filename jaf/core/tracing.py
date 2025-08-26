"""
Tracing and observability for the JAF framework.

This module provides tracing capabilities to monitor agent execution,
tool calls, and performance metrics.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

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

def create_composite_trace_collector(*collectors: TraceCollector) -> TraceCollector:
    """Create a composite trace collector that forwards events to multiple collectors."""
    
    collector_list = list(collectors)
    
    # Automatically add OTEL collector if URL is configured
    collector_url = os.getenv("TRACE_COLLECTOR_URL")
    if collector_url:
        setup_otel_tracing(collector_url=collector_url)
        otel_collector = OtelTraceCollector()
        collector_list.append(otel_collector)

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
