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

### Proxy Configuration

JAF supports proxy configuration for both Langfuse API calls and OpenTelemetry tracing.

**Langfuse Proxy Configuration**:

There are three ways to configure proxies for Langfuse:

1. **Environment Variables** (Recommended for production):
```bash
# Proxy configuration
export LANGFUSE_PROXY=http://proxy.company.com:8080

# Optional: Custom timeout (default: 10 seconds)
export LANGFUSE_TIMEOUT=30

# Standard HTTP proxy environment variables also work
# (if LANGFUSE_PROXY is not set, httpx will check these)
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080
```

2. **Direct Configuration**:
```python
from jaf.core.tracing import create_composite_trace_collector

# Pass proxy URL and timeout directly
trace_collector = create_composite_trace_collector(
    proxy="http://proxy.company.com:8080",
    timeout=30  # seconds
)
```

3. **Custom httpx Client** (For advanced configuration):
```python
import httpx
from jaf.core.tracing import LangfuseTraceCollector, create_composite_trace_collector

# Create httpx client with proxy, timeout, and custom headers
# Note: httpx proxy parameter accepts a string URL (applies to all protocols)
client = httpx.Client(
    proxy="http://proxy.company.com:8080",  # Same proxy for HTTP and HTTPS
    timeout=30.0,
    headers={"Custom-Header": "value"}
)

# Option 1: Direct instantiation
langfuse_collector = LangfuseTraceCollector(httpx_client=client)

# Option 2: Via composite collector
trace_collector = create_composite_trace_collector(httpx_client=client)
```

**Configuration Priority** (highest to lowest):
1. Custom `httpx_client` parameter (if provided, proxy/timeout params ignored)
2. Direct `proxy`/`timeout` parameters
3. `LANGFUSE_PROXY` and `LANGFUSE_TIMEOUT` environment variables
4. Standard `HTTP_PROXY`/`HTTPS_PROXY` environment variables (httpx default behavior)

**OpenTelemetry Proxy Configuration**:

OTEL supports proxy configuration in multiple ways:

1. **Environment Variables**:
```bash
# JAF-specific OTEL proxy (recommended)
export OTEL_PROXY=http://proxy.company.com:8080

# Standard HTTP proxy environment variables (fallback)
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080

# Optional: exclude certain hosts
export NO_PROXY=localhost,127.0.0.1
```

2. **Direct Configuration**:
```python
from jaf.core.tracing import setup_otel_tracing

# Pass proxy directly
setup_otel_tracing(
    collector_url="http://localhost:4318/v1/traces",
    proxy="http://proxy.company.com:8080",
    timeout=30  # seconds
)
```

3. **Custom requests.Session** (For advanced configuration):
```python
import requests
from jaf.core.tracing import setup_otel_tracing

# Create session with proxy and custom headers
session = requests.Session()
session.proxies = {
    'http': 'http://proxy.company.com:8080',
    'https': 'https://proxy.company.com:8080',
}
session.headers.update({'Custom-Header': 'value'})

setup_otel_tracing(
    collector_url="http://localhost:4318/v1/traces",
    session=session
)
```

4. **Via Composite Collector**:
```python
from jaf.core.tracing import create_composite_trace_collector

# Proxy applies to both OTEL and Langfuse
trace_collector = create_composite_trace_collector(
    proxy="http://proxy.company.com:8080",
    timeout=30
)

# Or with custom clients/sessions for each
import httpx
import requests

httpx_client = httpx.Client(proxy="http://proxy.company.com:8080")
requests_session = requests.Session()
requests_session.proxies = {'http': 'http://proxy.company.com:8080', 'https': 'https://proxy.company.com:8080'}
trace_collector = create_composite_trace_collector(
    httpx_client=httpx_client,  # For Langfuse
    otel_session=requests_session  # For OTEL
)
```

**Configuration Priority for OTEL** (highest to lowest):
1. Custom `session` parameter (if provided, proxy parameter ignored)
2. Direct `proxy` parameter
3. `OTEL_PROXY` environment variable
4. Standard `HTTP_PROXY`/`HTTPS_PROXY` environment variables

**Combined Setup** (Both Langfuse and OTEL with proxies):

Option 1: **Separate proxy configuration** (when using different proxies):
```bash
# Langfuse proxy and timeout
export LANGFUSE_PROXY=http://langfuse-proxy.company.com:8080
export LANGFUSE_TIMEOUT=30

# OTEL proxy
export OTEL_PROXY=http://otel-proxy.company.com:8080

# Langfuse credentials
export LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
export LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
export LANGFUSE_HOST=https://cloud.langfuse.com

# OTEL configuration
export TRACE_COLLECTOR_URL=http://localhost:4318/v1/traces
```

Option 2: **Single proxy for both** (most common):
```bash
# Single proxy for both services
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=https://proxy.company.com:8080

# Langfuse credentials
export LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
export LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
export LANGFUSE_HOST=https://cloud.langfuse.com

# OTEL configuration
export TRACE_COLLECTOR_URL=http://localhost:4318/v1/traces
```

Option 3: **Direct configuration in code**:
```python
from jaf.core.tracing import create_composite_trace_collector

# Single proxy for both Langfuse and OTEL
trace_collector = create_composite_trace_collector(
    proxy="http://proxy.company.com:8080",
    timeout=30
)

config = RunConfig(
    agent_registry={"agent": agent},
    model_provider=model_provider,
    on_event=trace_collector.collect
)
```

**Priority:** JAF-specific env vars (`LANGFUSE_PROXY`, `OTEL_PROXY`) take precedence over standard env vars (`HTTP_PROXY`, `HTTPS_PROXY`), allowing you to use different proxies for different services.

**Note:** If using `LangfuseTraceCollector` directly with proxy, ensure cleanup:
```python
langfuse_collector.close()  # Closes httpx client if owned
```

**Resource Cleanup**:

If you create a `LangfuseTraceCollector` with a proxy (not a custom httpx client), the collector manages an httpx client internally. This client is automatically closed when the collector is garbage collected, but you can also manually close it:

```python
from jaf.core.tracing import LangfuseTraceCollector

langfuse_collector = LangfuseTraceCollector(proxy="http://proxy.example.com:8080")

try:
    # Use the collector...
    pass
finally:
    # Explicitly close resources
    langfuse_collector.close()
```

For `create_composite_trace_collector()`, cleanup is handled automatically.

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

### Agent Name Tagging

JAF automatically tags all Langfuse traces with the agent name, enabling powerful filtering and analysis in the Langfuse dashboard. This feature provides enhanced observability for multi-agent systems.

**Automatic Tagging:**

Every trace in Langfuse includes the `agent_name` tag, allowing you to:
- Filter traces by specific agents
- Analyze performance per agent
- Track agent usage patterns
- Debug multi-agent workflows

**Example Dashboard Filtering:**

In your Langfuse dashboard, you can filter traces:
```
Tags: agent_name = "TechnicalSupport"
Tags: agent_name = "TriageAgent"
```

**Multi-Agent Analysis:**

```python
from jaf import Agent, RunConfig
from jaf.core.handoff import handoff_tool
from jaf.core.tracing import create_composite_trace_collector

# Create agents (each will be tagged separately)
triage_agent = Agent(
    name='TriageAgent',  # Tagged as "TriageAgent" in Langfuse
    instructions=lambda state: "Route users to specialists",
    tools=[handoff_tool],
    handoffs=['TechnicalSupport', 'Billing']
)

tech_support = Agent(
    name='TechnicalSupport',  # Tagged as "TechnicalSupport" in Langfuse
    instructions=lambda state: "Handle technical issues",
    tools=[debug_tool, restart_tool]
)

billing = Agent(
    name='Billing',  # Tagged as "Billing" in Langfuse
    instructions=lambda state: "Handle billing inquiries",
    tools=[invoice_tool, payment_tool]
)

# Set up tracing with Langfuse
trace_collector = create_composite_trace_collector()

config = RunConfig(
    agent_registry={
        'TriageAgent': triage_agent,
        'TechnicalSupport': tech_support,
        'Billing': billing
    },
    model_provider=model_provider,
    on_event=trace_collector.collect
)

# All traces will include agent_name tags automatically
result = await run(initial_state, config)
```

**Dashboard Analysis:**

In your Langfuse dashboard, you can now:

1. **Filter by Agent**: View traces for specific agents
2. **Compare Performance**: See which agents have higher latency or error rates
3. **Track Handoffs**: Follow conversations as they move between agents
4. **Optimize Costs**: Identify which agents consume the most tokens

**Viewing Agent Metrics:**

```
Dashboard → Traces → Filter by Tag: agent_name
- agent_name = "TriageAgent": 1,245 traces, avg latency 1.2s
- agent_name = "TechnicalSupport": 856 traces, avg latency 2.5s
- agent_name = "Billing": 623 traces, avg latency 1.8s
```

This automatic tagging works seamlessly with JAF's handoff system, allowing you to trace the complete journey of a user conversation across multiple specialized agents.

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

## Distributed Tracing and Custom IDs

### Overview

JAF supports distributed tracing scenarios where you need to track conversations and agent executions across multiple deployments, services, or agent hierarchies. Key features include:

- **Automatic Trace Propagation**: Sub-agents automatically inherit the parent's `trace_id`
- **Custom Trace IDs**: Provide your own `trace_id` to link traces across distributed systems
- **Custom Session IDs**: Use `conversation_id` to maintain session continuity
- **Multi-Level Hierarchies**: Unified tracing across deep agent hierarchies

### Automatic Trace Propagation to Sub-Agents

When using agent-as-tool (sub-agents), the parent's `trace_id` is automatically inherited by all child agents, creating a unified trace:

```python
from jaf import Agent, RunState, RunConfig
from jaf.core.agent_tool import create_agent_tool
from jaf.core.types import TraceId, generate_run_id

# Create a sub-agent
research_agent = Agent(
    name="research_specialist",
    instructions=lambda s: "You are a research specialist.",
    tools=[research_tool]
)

# Convert to tool
research_tool = create_agent_tool(
    agent=research_agent,
    tool_name="delegate_research"
)

# Main agent uses the sub-agent
main_agent = Agent(
    name="orchestrator",
    instructions=lambda s: "Delegate research tasks.",
    tools=[research_tool]
)

# Create initial state with trace_id
initial_state = RunState(
    run_id=generate_run_id(),
    trace_id=TraceId("trace_parent_123"),  # Parent trace_id
    messages=[...],
    current_agent_name="orchestrator",
    context={},
    turn_count=0
)

# Run the agent
result = await run(initial_state, config)

# Both orchestrator and research_specialist will use "trace_parent_123"
# All events appear under one unified trace in your tracing backend!
```

### Custom Trace IDs for Distributed Systems

Provide your own `trace_id` to link agent executions across multiple deployments or services:

```python
from jaf.core.types import TraceId, generate_run_id

# Receive trace_id from upstream service (e.g., HTTP header, message queue)
upstream_trace_id = "trace_from_api_gateway_abc123"

# Service #1: Initial processing
initial_state = RunState(
    run_id=generate_run_id(),
    trace_id=TraceId(upstream_trace_id),  # Use the upstream trace_id
    messages=[Message(role=ContentRole.USER, content="Request from service 1")],
    current_agent_name="service_1_agent",
    context={},
    turn_count=0
)

result_1 = await run(initial_state, config)

# Service #2: Continue processing with SAME trace_id
# (perhaps in a different deployment/container)
followup_state = RunState(
    run_id=generate_run_id(),  # New run_id for this service
    trace_id=TraceId(upstream_trace_id),  # SAME trace_id for unified tracing
    messages=[Message(role=ContentRole.USER, content="Request from service 2")],
    current_agent_name="service_2_agent",
    context={},
    turn_count=0
)

result_2 = await run(followup_state, config)

# Both executions are linked in your tracing backend under the same trace_id!
```

### Custom Session IDs

Use `conversation_id` in `RunConfig` to provide a custom session ID:

```python
# Receive session_id from upstream (e.g., user session from auth service)
user_session_id = "session_user_456_from_auth"

config = RunConfig(
    agent_registry={"agent": agent},
    model_provider=model_provider,
    conversation_id=user_session_id,  # Custom session_id
    on_event=trace_collector.collect
)

# The session_id will be included in all trace events
result = await run(initial_state, config)
```

### Session Continuity with preserve_session

Control whether sub-agents share the parent's session (conversation_id) or get isolated sessions:

```python
from jaf.core.agent_tool import create_agent_tool

# Option 1: Preserve session (shared memory/conversation history)
specialist_tool_shared = create_agent_tool(
    agent=specialist_agent,
    tool_name="call_specialist_shared",
    preserve_session=True  # Sub-agent inherits parent's conversation_id
)

# Option 2: Isolated session (ephemeral, per-invocation)
specialist_tool_isolated = create_agent_tool(
    agent=specialist_agent,
    tool_name="call_specialist_isolated",
    preserve_session=False  # Default: each call gets its own session
)
```

**When to use `preserve_session=True`:**
- Sub-agent needs access to conversation history
- Shared memory across agent calls
- Continuous conversation flow

**When to use `preserve_session=False` (default):**
- Independent, stateless sub-agent calls
- No memory sharing needed
- Each invocation should be isolated

### Multi-Level Agent Hierarchies

Trace IDs propagate through any depth of agent hierarchy:

```python
# Level 3: Worker agent
worker = Agent(name="worker", ...)
worker_tool = create_agent_tool(agent=worker)

# Level 2: Manager agent (calls worker)
manager = Agent(name="manager", tools=[worker_tool], ...)
manager_tool = create_agent_tool(agent=manager)

# Level 1: Main agent (calls manager)
main = Agent(name="main", tools=[manager_tool], ...)

# All three levels will share the same trace_id
initial_state = RunState(
    trace_id=TraceId("trace_hierarchy_123"),
    ...
)

result = await run(initial_state, config)

# View the entire call hierarchy under "trace_hierarchy_123" in your tracing backend!
```

### Use Cases

**Distributed Microservices:**
```python
# API Gateway receives request
trace_id = request.headers.get("X-Trace-Id") or generate_trace_id()
session_id = request.headers.get("X-Session-Id")

# Service A
config_a = RunConfig(conversation_id=session_id, ...)
state_a = RunState(trace_id=TraceId(trace_id), ...)
await run(state_a, config_a)

# Service B (different deployment)
config_b = RunConfig(conversation_id=session_id, ...)
state_b = RunState(trace_id=TraceId(trace_id), ...)
await run(state_b, config_b)

# Both services' traces are unified under the same trace_id!
```

**Multi-Tenant Systems:**
```python
# Use tenant_id as session_id for tenant-specific tracing
tenant_id = "tenant_acme_corp"

config = RunConfig(
    conversation_id=tenant_id,  # All events tagged with tenant_id
    ...
)
```

**Cross-Service Agent Calls:**
```python
# Service 1: Main agent
trace_id = TraceId("trace_cross_service_123")
state = RunState(trace_id=trace_id, ...)
result = await run(state, config)

# Pass trace_id to Service 2 via HTTP header, message queue, etc.
response = requests.post(
    "https://service-2/agent",
    headers={"X-Trace-Id": str(trace_id)}
)

# Service 2: Continues with same trace_id
received_trace_id = request.headers["X-Trace-Id"]
state_2 = RunState(trace_id=TraceId(received_trace_id), ...)
result_2 = await run(state_2, config_2)
```

### Complete Example

See `examples/distributed_tracing_example.py` for comprehensive examples demonstrating:

1. Automatic trace propagation to sub-agents
2. Custom trace IDs for distributed systems
3. Multi-level agent hierarchies
4. Session continuity patterns

```bash
# Run the example
python examples/distributed_tracing_example.py
```

### Best Practices

1. **Trace ID Generation**:
   - Generate trace IDs at system entry points (API gateways, message queues)
   - Pass trace IDs through all service calls
   - Use meaningful prefixes: `trace_api_`, `trace_queue_`, etc.

2. **Session ID Management**:
   - Link session IDs to user sessions or logical groupings
   - Use session IDs for filtering in tracing dashboards
   - Document session ID lifecycle

3. **Sub-Agent Configuration**:
   - Use `preserve_session=True` sparingly (when memory sharing is needed)
   - Default to isolated sessions for better agent independence
   - Document which agent tools preserve sessions

4. **Monitoring**:
   - Monitor trace ID propagation in your tracing backend
   - Alert on broken trace chains (missing trace IDs)
   - Track session duration and boundaries

### Tracing Backend Integration

Custom trace IDs and session IDs work seamlessly with all JAF tracing backends:

**Langfuse:**
```python
# Traces are automatically grouped by trace_id
# Session_id appears in trace metadata
# Agent names are tagged for filtering
```

**OpenTelemetry:**
```python
# Trace IDs map to OpenTelemetry trace context
# Session IDs appear as span attributes
# Complete distributed tracing support
```

**Custom Collectors:**
```python
class CustomCollector:
    def collect(self, event: TraceEvent):
        trace_id = event.data.get("trace_id")
        session_id = event.data.get("session_id")
        # Index by trace_id and session_id for querying
```

This comprehensive tracing system enables full observability of your JAF agents in any environment, from development to production.