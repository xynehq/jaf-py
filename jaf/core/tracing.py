"""
Tracing and observability for the JAF framework.

This module provides tracing capabilities to monitor agent execution,
tool calls, and performance metrics.
"""

import json
import time
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from .types import TraceEvent

class TraceCollector(Protocol):
    """Protocol for trace collectors."""
    
    def collect(self, event: TraceEvent) -> None:
        """Collect a trace event."""
        ...

class ConsoleTraceCollector:
    """Simple trace collector that outputs to console."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def collect(self, event: TraceEvent) -> None:
        """Collect and log a trace event to console."""
        timestamp = datetime.now().isoformat()
        event_dict = {
            'timestamp': timestamp,
            'type': event.type,
            'data': event.data
        }
        
        self.events.append(event_dict)
        
        if self.verbose:
            self._log_event(event_dict)
    
    def _log_event(self, event: Dict[str, Any]) -> None:
        """Log an event to the console with formatting."""
        event_type = event['type']
        data = event['data']
        
        if event_type == 'run_start':
            print(f"🚀 [TRACE] Run started - Run ID: {data.get('run_id')}, Trace ID: {data.get('trace_id')}")
        
        elif event_type == 'llm_call_start':
            print(f"🤖 [TRACE] LLM call started - Agent: {data.get('agent_name')}, Model: {data.get('model')}")
        
        elif event_type == 'llm_call_end':
            print(f"✅ [TRACE] LLM call completed")
        
        elif event_type == 'tool_call_start':
            print(f"🔧 [TRACE] Tool call started - Tool: {data.get('tool_name')}")
            if data.get('args'):
                print(f"   Args: {json.dumps(data['args'], indent=2)}")
        
        elif event_type == 'tool_call_end':
            tool_name = data.get('tool_name')
            status = data.get('status', 'unknown')
            print(f"🛠️  [TRACE] Tool call completed - Tool: {tool_name}, Status: {status}")
            
            # Show result summary (truncated if too long)
            result = data.get('result', '')
            if isinstance(result, str) and len(result) > 200:
                result = result[:200] + "..."
            print(f"   Result: {result}")
        
        elif event_type == 'handoff':
            print(f"🔄 [TRACE] Agent handoff - From: {data.get('from')} → To: {data.get('to')}")
        
        elif event_type == 'run_end':
            outcome = data.get('outcome')
            elapsed = time.time() - self.start_time
            
            if outcome and hasattr(outcome, 'status'):
                status = outcome.status
                
                if status == 'completed':
                    print(f"🎉 [TRACE] Run completed successfully in {elapsed:.2f}s")
                else:
                    error = outcome.error if hasattr(outcome, 'error') else None
                    error_type = getattr(error, '_tag', 'unknown') if error and hasattr(error, '_tag') else 'unknown'
                    print(f"❌ [TRACE] Run failed with {error_type} in {elapsed:.2f}s")
            else:
                print(f"🎉 [TRACE] Run completed in {elapsed:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected events."""
        total_events = len(self.events)
        event_counts = {}
        
        for event in self.events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate durations
        run_start = next((e for e in self.events if e['type'] == 'run_start'), None)
        run_end = next((e for e in self.events if e['type'] == 'run_end'), None)
        
        duration = None
        if run_start and run_end:
            start_time = datetime.fromisoformat(run_start['timestamp'])
            end_time = datetime.fromisoformat(run_end['timestamp'])
            duration = (end_time - start_time).total_seconds()
        
        return {
            'total_events': total_events,
            'event_counts': event_counts,
            'duration_seconds': duration,
            'events': self.events
        }
    
    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()
        self.start_time = time.time()

class FileTraceCollector:
    """Trace collector that writes to a file."""
    
    def __init__(self, file_path: str, pretty_print: bool = True):
        self.file_path = file_path
        self.pretty_print = pretty_print
        self.events: List[Dict[str, Any]] = []
    
    def collect(self, event: TraceEvent) -> None:
        """Collect a trace event and write to file."""
        timestamp = datetime.now().isoformat()
        event_dict = {
            'timestamp': timestamp,
            'type': event.type,
            'data': event.data
        }
        
        self.events.append(event_dict)
        self._write_to_file(event_dict)
    
    def _write_to_file(self, event: Dict[str, Any]) -> None:
        """Write an event to the trace file."""
        try:
            with open(self.file_path, 'a') as f:
                if self.pretty_print:
                    f.write(json.dumps(event, indent=2, default=str) + '\n')
                else:
                    f.write(json.dumps(event, default=str) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write trace to file {self.file_path}: {e}")

class MemoryTraceCollector:
    """Trace collector that stores events in memory for analysis."""
    
    def __init__(self, max_events: Optional[int] = None):
        self.max_events = max_events
        self.events: List[Dict[str, Any]] = []
    
    def collect(self, event: TraceEvent) -> None:
        """Collect a trace event in memory."""
        timestamp = datetime.now().isoformat()
        event_dict = {
            'timestamp': timestamp,
            'type': event.type,
            'data': event.data
        }
        
        self.events.append(event_dict)
        
        # Trim events if we exceed max_events
        if self.max_events and len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get collected events, optionally filtered by type."""
        if event_type:
            return [e for e in self.events if e['type'] == event_type]
        return self.events.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from collected events."""
        tool_calls = [e for e in self.events if e['type'] == 'tool_call_end']
        llm_calls = [e for e in self.events if e['type'] == 'llm_call_end']
        
        # Calculate tool execution times
        tool_metrics = {}
        for event in tool_calls:
            tool_name = event['data'].get('tool_name')
            if tool_name:
                if tool_name not in tool_metrics:
                    tool_metrics[tool_name] = []
                
                # Look for execution time in metadata
                tool_result = event['data'].get('tool_result', {})
                if isinstance(tool_result, dict):
                    metadata = tool_result.get('metadata', {})
                    exec_time = metadata.get('execution_time_ms')
                    if exec_time:
                        tool_metrics[tool_name].append(exec_time)
        
        # Calculate averages
        tool_averages = {}
        for tool_name, times in tool_metrics.items():
            if times:
                tool_averages[tool_name] = {
                    'avg_execution_time_ms': sum(times) / len(times),
                    'min_execution_time_ms': min(times),
                    'max_execution_time_ms': max(times),
                    'call_count': len(times)
                }
        
        return {
            'total_tool_calls': len(tool_calls),
            'total_llm_calls': len(llm_calls),
            'tool_metrics': tool_averages
        }
    
    def clear(self) -> None:
        """Clear collected events."""
        self.events.clear()

class CompositeTraceCollector:
    """Trace collector that forwards events to multiple collectors."""
    
    def __init__(self, collectors: List[TraceCollector]):
        self.collectors = collectors
    
    def collect(self, event: TraceEvent) -> None:
        """Forward event to all collectors."""
        for collector in self.collectors:
            try:
                collector.collect(event)
            except Exception as e:
                print(f"Warning: Trace collector failed: {e}")
    
    def add_collector(self, collector: TraceCollector) -> None:
        """Add a new collector."""
        self.collectors.append(collector)
    
    def remove_collector(self, collector: TraceCollector) -> None:
        """Remove a collector."""
        if collector in self.collectors:
            self.collectors.remove(collector)