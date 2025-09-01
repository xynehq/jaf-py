# Tracing and Observability

JAF provides comprehensive tracing and observability capabilities to monitor agent execution, performance metrics, and system behavior. This guide covers all available tracing options including OpenTelemetry, Langfuse, and custom collectors.

## Overview

JAF's tracing system follows a publisher-subscriber pattern where the core execution engine emits trace events, and various collectors consume these events for monitoring, debugging, and analytics.

### Key Features

- **Multiple Trace Backends**: OpenTelemetry, Langfuse, console, file, and in-memory collectors
- **Automatic Configuration**: Environment-based setup with sensible defaults
- **Composite Collectors**: Combine multiple collectors for comprehensive observability
- **Event-Driven Architecture**: Minimal performance overhead with async event handling
- **Production Ready**: Designed for high-throughput production environments

### Trace Events

JAF emits the following trace events during execution:

- `run_start` - Agent run initialization
- `run_end` - Agent run completion with outcome
- `llm_call_start` - LLM request initiated
- `llm_call_end` - LLM response received
- `tool_call_start` - Tool execution started
- `tool_call_end` - Tool execution completed
- `handoff` - Agent handoff occurred
- `error` - Error conditions and failures

## Quick Start

### Basic Console Tracing

For development and debugging, start with console tracing:

```python
from jaf import run, RunConfig, RunState
from jaf.core.tracing import ConsoleTraceCollector

# Create console trace collector
trace_collector = ConsoleTraceCollector()

# Configure with tracing
config = RunConfig(
    agent_registry={"my_agent": agent},
    model_provider=model_provider,
    on_event=trace_collector.collect  # Enable tracing
)

# Run with tracing enabled
result = await run(initial_state, config)
```

### Auto-Configuration

JAF automatically configures tracing based on environment variables:

```python
from jaf.core.tracing import create_composite_trace_collector

# Automatically includes enabled collectors based on environment
trace_collector = create_composite_trace_collector()

config = RunConfig(
    agent_registry={"my_agent": agent},
    model_provider=model_provider,
    on_event=trace_collector.collect
)
```

Environment variables for auto-configuration:

```bash
# Enable OpenTelemetry (requires TRACE_COLLECTOR_URL)
TRACE_COLLECTOR_URL=http://localhost:4318/v1/traces

# Enable Langfuse (requires both keys)
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, defaults to cloud
```

## OpenTelemetry Integration

### Setup and Configuration

JAF integrates with OpenTelemetry for industry-standard observability:

```python
import os
from jaf.core.tracing import setup_otel_tracing, OtelTraceCollector

# Set environment variable for auto-configuration
os.environ["TRACE_COLLECTOR_URL"] = "http://localhost:4318/v1/traces"

# Manual setup (optional)
setup_otel_tracing(
    service_name="jaf-agent",
    collector_url="http://localhost:4318/v1/traces"
)

# Create OTEL collector
otel_collector = OtelTraceCollector(service_name="my-jaf-service")
```

### Running with Jaeger

Set up Jaeger for OpenTelemetry traces:

```bash
# Start Jaeger all-in-one
docker run -d \
  --name jaeger \
  -p 16686:16686 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Set environment for JAF
export TRACE_COLLECTOR_URL=http://localhost:4318/v1/traces
```

### Complete OpenTelemetry Example

```python
import asyncio
import os
from jaf import Agent, Message, ModelConfig, RunConfig, RunState
from jaf.core.engine import run
from jaf.core.types import ContentRole, generate_run_id, generate_trace_id
from jaf.core.tracing import ConsoleTraceCollector, create_composite_trace_collector
from jaf.providers.model import make_litellm_provider

# Configure OpenTelemetry
os.environ["TRACE_COLLECTOR_URL"] = "http://localhost:4318/v1/traces"

async def main():
    # Create agent with tools
    agent = Agent(
        name="demo_agent",
        instructions=lambda s: "You are a helpful assistant.",
        model_config=ModelConfig(name="gpt-4")
    )

    # Auto-configured tracing (includes OTEL + Console)
    trace_collector = create_composite_trace_collector(ConsoleTraceCollector())

    config = RunConfig(
        agent_registry={"demo_agent": agent},
        model_provider=make_litellm_provider(
            base_url="http://localhost:4000",
            api_key="your-api-key"
        ),
        on_event=trace_collector.collect
    )

    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content="Hello!")],
        current_agent_name="demo_agent",
        context={},
        turn_count=0
    )

    result = await run(initial_state, config)
    print(f"Result: {result.outcome}")

if __name__ == "__main__":
    asyncio.run(main())
```

View traces at [http://localhost:16686](http://localhost:16686)

### Production OpenTelemetry Setup

For production environments with OTLP exporters:

```python
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Production OTEL setup
def setup_production_tracing():
    resource = Resource.create({
        "service.name": "jaf-agent-prod",
        "service.version": "2.2.2",
        "deployment.environment": os.getenv("ENVIRONMENT", "production")
    })
    
    provider = TracerProvider(resource=resource)
    
    # OTLP gRPC exporter for production
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
        headers={
            "authorization": f"Bearer {os.getenv('OTEL_EXPORTER_OTLP_HEADERS_AUTHORIZATION')}"
        }
    )
    
    # Batch processor for performance
    span_processor = BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=2048,
        export_timeout_millis=30000,
        max_export_batch_size=512
    )
    
    provider.add_span_processor(span_processor)
    trace.set_tracer_provider(provider)

# Use in production
setup_production_tracing()
```

## Langfuse Integration

### Setup and Configuration

JAF integrates with Langfuse for advanced LLM observability:

```python
import os
from jaf.core.tracing import LangfuseTraceCollector

# Set environment variables
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-your-public-key"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-your-secret-key"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # Optional

# Manual collector creation
langfuse_collector = LangfuseTraceCollector()
```

### Langfuse Cloud Setup

1. **Create Account**: Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. **Create Project**: Create a new project in your dashboard
3. **Get API Keys**: Copy public key and secret key from project settings
4. **Configure Environment**:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export LANGFUSE_HOST=https://cloud.langfuse.com
```

### Self-Hosted Langfuse

For self-hosted Langfuse instances:

```bash
# Start Langfuse with Docker
docker run -d \
  --name langfuse \
  -p 3000:3000 \
  -e DATABASE_URL=postgresql://user:password@db:5432/langfuse \
  -e NEXTAUTH_SECRET=your-secret-key \
  -e SALT=your-salt \
  langfuse/langfuse:latest

# Configure JAF
export LANGFUSE_PUBLIC_KEY=pk-lf-your-local-key
export LANGFUSE_SECRET_KEY=sk-lf-your-local-secret
export LANGFUSE_HOST=http://localhost:3000
```

### Complete Langfuse Example

```python
import asyncio
import os
from typing import Annotated, Literal
from pydantic import BaseModel, Field

from jaf import Agent, Message, ModelConfig, RunConfig, RunState
from jaf.core.engine import run
from jaf.core.types import ContentRole, generate_run_id, generate_trace_id
from jaf.core.tools import create_function_tool
from jaf.core.tracing import ConsoleTraceCollector, create_composite_trace_collector
from jaf.providers.model import make_litellm_provider

# Configure Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-your-public-key"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-your-secret-key"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

class Weather(BaseModel):
    location: str = Field(..., description="Location to get weather for")
    unit: Annotated[Literal["celsius", "fahrenheit"], "Temperature unit"] = "celsius"

async def get_weather(args: Weather, context) -> str:
    """Get weather for a location."""
    if "new york" in args.location.lower():
        return f"Weather in New York: 75°{args.unit}"
    return f"Weather in {args.location}: 25°{args.unit}"

async def main():
    # Create weather tool
    weather_tool = create_function_tool({
        "name": "get_weather",
        "description": "Get current weather for a location",
        "execute": get_weather,
        "parameters": Weather,
    })

    # Create agent with tools
    agent = Agent(
        name="weather_agent",
        instructions=lambda s: "You are a weather assistant. Use the weather tool to answer questions.",
        tools=[weather_tool],
        model_config=ModelConfig(name="gpt-4")
    )

    # Auto-configured tracing (includes Langfuse + Console)
    trace_collector = create_composite_trace_collector(ConsoleTraceCollector())

    config = RunConfig(
        agent_registry={"weather_agent": agent},
        model_provider=make_litellm_provider(
            base_url="http://localhost:4000",
            api_key="your-api-key"
        ),
        on_event=trace_collector.collect
    )

    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content="What's the weather in New York?")],
        current_agent_name="weather_agent",
        context={"user_id": "user-123", "session_id": "session-456"},
        turn_count=0
    )

    result = await run(initial_state, config)
    
    if result.outcome.status == "completed":
        print(f"Final result: {result.outcome.output}")
    else:
        print(f"Error: {result.outcome.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

## File and In-Memory Collectors

### File Trace Collector

For persistent trace storage and analysis:

```python
from jaf.core.tracing import FileTraceCollector

# Create file collector
file_collector = FileTraceCollector("traces/agent_traces.jsonl")

config = RunConfig(
    agent_registry={"agent": agent},
    model_provider=model_provider,
    on_event=file_collector.collect
)

# Traces written to traces/agent_traces.jsonl as JSONL
```

Example trace file format:

```json
{"timestamp": "2024-01-15T10:30:00.123Z", "type": "run_start", "data": {"run_id": "run_123", "trace_id": "trace_456"}}
{"timestamp": "2024-01-15T10:30:01.456Z", "type": "llm_call_start", "data": {"model": "gpt-4", "agent_name": "weather_agent"}}
{"timestamp": "2024-01-15T10:30:02.789Z", "type": "llm_call_end", "data": {"choice": {"message": {"content": "I'll help you with the weather."}}}}
```

### In-Memory Trace Collector

For testing and development:

```python
from jaf.core.tracing import InMemoryTraceCollector

# Create in-memory collector
memory_collector = InMemoryTraceCollector()

config = RunConfig(
    agent_registry={"agent": agent},
    model_provider=model_provider,
    on_event=memory_collector.collect
)

# After execution, retrieve traces
all_traces = memory_collector.get_all_traces()
specific_trace = memory_collector.get_trace("trace_id_123")

# Clear traces when needed
memory_collector.clear()  # Clear all
memory_collector.clear("trace_id_123")  # Clear specific trace
```

## Custom Trace Collectors

### Implementing Custom Collectors

Create custom collectors for specialized observability needs:

```python
from typing import Dict, List, Optional
from jaf.core.types import TraceEvent, TraceId
from jaf.core.tracing import TraceCollector

class MetricsTraceCollector:
    """Custom collector that tracks performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_llm_calls": 0,
            "total_tool_calls": 0,
            "avg_run_duration": 0.0
        }
        self.run_start_times = {}
    
    def collect(self, event: TraceEvent) -> None:
        """Collect metrics from trace events."""
        if event.type == "run_start":
            self.metrics["total_runs"] += 1
            run_id = event.data.get("run_id")
            if run_id:
                self.run_start_times[run_id] = event.data.get("timestamp", 0)
        
        elif event.type == "run_end":
            outcome = event.data.get("outcome")
            if outcome and hasattr(outcome, "status"):
                if outcome.status == "completed":
                    self.metrics["successful_runs"] += 1
                else:
                    self.metrics["failed_runs"] += 1
            
            # Calculate duration
            run_id = event.data.get("run_id")
            if run_id and run_id in self.run_start_times:
                start_time = self.run_start_times[run_id]
                end_time = event.data.get("timestamp", 0)
                duration = end_time - start_time
                
                # Update average duration
                total_completed = self.metrics["successful_runs"] + self.metrics["failed_runs"]
                if total_completed > 0:
                    current_avg = self.metrics["avg_run_duration"]
                    self.metrics["avg_run_duration"] = (
                        (current_avg * (total_completed - 1) + duration) / total_completed
                    )
                
                del self.run_start_times[run_id]
        
        elif event.type == "llm_call_start":
            self.metrics["total_llm_calls"] += 1
        
        elif event.type == "tool_call_start":
            self.metrics["total_tool_calls"] += 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {key: 0 if isinstance(value, (int, float)) else value 
                       for key, value in self.metrics.items()}
        self.run_start_times.clear()

# Usage
metrics_collector = MetricsTraceCollector()

config = RunConfig(
    agent_registry={"agent": agent},
    model_provider=model_provider,
    on_event=metrics_collector.collect
)

# After some runs
print("Performance Metrics:", metrics_collector.get_metrics())
```

### Database Trace Collector

For enterprise observability with database storage:

```python
import asyncio
import json
from datetime import datetime
import asyncpg

class PostgreSQLTraceCollector:
    """Trace collector that stores events in PostgreSQL."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def init_pool(self):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        # Create traces table if not exists
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_traces (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    trace_id VARCHAR(255),
                    run_id VARCHAR(255),
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON agent_traces(trace_id);
                CREATE INDEX IF NOT EXISTS idx_traces_run_id ON agent_traces(run_id);
                CREATE INDEX IF NOT EXISTS idx_traces_event_type ON agent_traces(event_type);
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON agent_traces(timestamp);
            """)
    
    def collect(self, event: TraceEvent) -> None:
        """Collect trace event (async wrapper)."""
        asyncio.create_task(self._async_collect(event))
    
    async def _async_collect(self, event: TraceEvent) -> None:
        """Asynchronously store trace event."""
        if not self.pool:
            await self.init_pool()
        
        # Extract trace and run IDs
        trace_id = None
        run_id = None
        
        if event.data:
            trace_id = event.data.get("trace_id") or event.data.get("traceId")
            run_id = event.data.get("run_id") or event.data.get("runId")
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_traces (timestamp, trace_id, run_id, event_type, event_data)
                VALUES ($1, $2, $3, $4, $5)
            """, datetime.utcnow(), trace_id, run_id, event.type, json.dumps(event.data, default=str))
    
    async def get_trace_events(self, trace_id: str) -> List[Dict]:
        """Get all events for a trace."""
        if not self.pool:
            await self.init_pool()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT timestamp, event_type, event_data 
                FROM agent_traces 
                WHERE trace_id = $1 
                ORDER BY timestamp
            """, trace_id)
            
            return [dict(row) for row in rows]
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

# Usage
async def main():
    db_collector = PostgreSQLTraceCollector("postgresql://user:pass@localhost/traces")
    
    config = RunConfig(
        agent_registry={"agent": agent},
        model_provider=model_provider,
        on_event=db_collector.collect
    )
    
    # Run agents...
    
    # Query traces
    events = await db_collector.get_trace_events("trace_123")
    print(f"Found {len(events)} events for trace")
    
    await db_collector.close()
```

## Composite Collectors

### Combining Multiple Collectors

Use composite collectors for comprehensive observability:

```python
from jaf.core.tracing import create_composite_trace_collector, ConsoleTraceCollector, FileTraceCollector

# Manual composition
console_collector = ConsoleTraceCollector()
file_collector = FileTraceCollector("traces/production.jsonl")
metrics_collector = MetricsTraceCollector()

composite_collector = create_composite_trace_collector(
    console_collector,
    file_collector,
    metrics_collector
)

# Auto-composition with environment variables
# This will automatically include OTEL and Langfuse if configured
auto_collector = create_composite_trace_collector(
    ConsoleTraceCollector(),  # Always include console for development
    metrics_collector         # Add custom metrics
)

config = RunConfig(
    agent_registry={"agent": agent},
    model_provider=model_provider,
    on_event=composite_collector.collect
)
```

### Error Handling in Collectors

Composite collectors handle individual collector failures gracefully:

```python
# If one collector fails, others continue working
# Errors are logged but don't stop execution
composite_collector = create_composite_trace_collector(
    ConsoleTraceCollector(),           # Always works
    FileTraceCollector("/read-only/"),  # Might fail
    LangfuseTraceCollector()           # Might have network issues
)

# Failed collectors log warnings but don't crash the application
```

## Production Deployment

### Environment Configuration

Production environment setup for comprehensive tracing:

```bash
# Production environment variables
export ENVIRONMENT=production
export SERVICE_NAME=jaf-agent-prod
export SERVICE_VERSION=2.2.2

# OpenTelemetry Configuration
export TRACE_COLLECTOR_URL=https://otlp-gateway.company.com/v1/traces
export OTEL_EXPORTER_OTLP_HEADERS_AUTHORIZATION=Bearer your-token

# Langfuse Configuration
export LANGFUSE_PUBLIC_KEY=pk-lf-production-key
export LANGFUSE_SECRET_KEY=sk-lf-production-secret
export LANGFUSE_HOST=https://langfuse.company.com

# Performance Settings
export JAF_TRACE_BUFFER_SIZE=1000
export JAF_TRACE_FLUSH_INTERVAL=30
export JAF_TRACE_ENABLED=true
```

### Production Trace Setup

```python
import os
from jaf.core.tracing import create_composite_trace_collector, FileTraceCollector

def create_production_tracing():
    """Create production-ready tracing configuration."""
    collectors = []
    
    # File collector for local debugging
    if os.getenv("JAF_TRACE_FILE_ENABLED", "false").lower() == "true":
        trace_file = os.getenv("JAF_TRACE_FILE", "/var/log/jaf/traces.jsonl")
        collectors.append(FileTraceCollector(trace_file))
    
    # Custom metrics collector
    if os.getenv("JAF_METRICS_ENABLED", "true").lower() == "true":
        collectors.append(MetricsTraceCollector())
    
    # Auto-includes OTEL and Langfuse based on environment
    return create_composite_trace_collector(*collectors)

# Production usage
trace_collector = create_production_tracing()

config = RunConfig(
    agent_registry=agents,
    model_provider=model_provider,
    on_event=trace_collector.collect
)
```

### Performance Considerations

1. **Async Collection**: All collectors should be async-friendly
2. **Buffering**: Use batched exports for high-throughput scenarios
3. **Sampling**: Consider trace sampling for very high volume
4. **Error Isolation**: Failed collectors shouldn't affect others
5. **Resource Limits**: Set appropriate buffer sizes and timeouts

### Monitoring and Alerting

Set up monitoring based on trace data:

```python
class AlertingTraceCollector:
    """Collector that sends alerts on errors."""
    
    def __init__(self, webhook_url: str, error_threshold: int = 5):
        self.webhook_url = webhook_url
        self.error_threshold = error_threshold
        self.error_count = 0
    
    def collect(self, event: TraceEvent) -> None:
        if event.type == "error" or (
            event.type == "run_end" and 
            event.data.get("outcome", {}).get("status") == "failed"
        ):
            self.error_count += 1
            if self.error_count >= self.error_threshold:
                self.send_alert(event)
                self.error_count = 0  # Reset counter
    
    def send_alert(self, event: TraceEvent):
        """Send alert webhook."""
        # Implementation for sending alerts
        pass
```

## Best Practices

### Development

1. **Use Console Tracing**: Always include console tracing during development
2. **File Backup**: Save traces to files for later analysis
3. **Test Collectors**: Verify custom collectors work correctly
4. **Environment Separation**: Use different trace configurations per environment

### Production

1. **Multiple Backends**: Use composite collectors for redundancy
2. **Error Handling**: Ensure trace failures don't affect agent execution
3. **Performance**: Monitor trace collector performance and resource usage
4. **Data Retention**: Implement appropriate trace data retention policies
5. **Security**: Protect sensitive data in trace events

### Debugging

1. **Trace IDs**: Use consistent trace IDs across your system
2. **Event Correlation**: Correlate trace events with application logs
3. **Time Synchronization**: Ensure accurate timestamps across collectors
4. **Structured Data**: Use structured event data for better analysis

## Troubleshooting

### Common Issues

**No traces appearing in OpenTelemetry**:
- Verify `TRACE_COLLECTOR_URL` is set correctly
- Check if OTLP collector is running and accessible
- Ensure network connectivity to the collector endpoint

**Langfuse authentication errors**:
- Verify API keys are correct and not expired
- Check if the Langfuse host URL is accessible
- Ensure proper environment variable names

**High memory usage with tracing**:
- Use appropriate buffer sizes for collectors
- Implement trace sampling for high volume
- Monitor collector resource usage

**Trace events missing**:
- Verify `on_event` is set in RunConfig
- Check if collectors are properly initialized
- Look for error messages in collector logs

### Debug Mode

Enable debug mode for detailed tracing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from jaf.core.tracing import ConsoleTraceCollector

# Console collector includes detailed debug output
debug_collector = ConsoleTraceCollector()
```

This comprehensive tracing system enables full observability of your JAF agents in any environment, from development to production.