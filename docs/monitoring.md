# Monitoring and Observability

Comprehensive monitoring, logging, and observability setup for JAF applications in production.

## Overview

Effective monitoring is crucial for maintaining reliable JAF deployments. This guide covers metrics collection, logging strategies, alerting, and observability best practices.

## Metrics Collection

### Prometheus Integration

JAF provides built-in Prometheus metrics for comprehensive monitoring:

```python
from jaf import create_metrics_collector
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Initialize metrics
AGENT_REQUESTS = Counter('jaf_agent_requests_total', 'Total agent requests', ['agent_name', 'status'])
AGENT_DURATION = Histogram('jaf_agent_request_duration_seconds', 'Agent request duration', ['agent_name'])
ACTIVE_CONVERSATIONS = Gauge('jaf_active_conversations', 'Number of active conversations')
MEMORY_USAGE = Gauge('jaf_memory_usage_bytes', 'Memory usage by component', ['component'])

# Start metrics server
start_http_server(9090)

# Collect metrics in your agent
class MonitoredAgent:
    async def process_message(self, message, context):
        start_time = time.time()
        status = 'success'
        
        try:
            result = await self.agent.process(message, context)
            return result
        except Exception as e:
            status = 'error'
            raise
        finally:
            AGENT_REQUESTS.labels(agent_name=self.name, status=status).inc()
            AGENT_DURATION.labels(agent_name=self.name).observe(time.time() - start_time)
```

### Key Metrics to Monitor

#### Performance Metrics
```python
# Response time percentiles
RESPONSE_TIME = Histogram(
    'jaf_response_time_seconds',
    'Agent response time',
    ['agent_name', 'tool_name'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
)

# Throughput
REQUEST_RATE = Counter(
    'jaf_requests_per_second',
    'Request rate',
    ['agent_name']
)

# Error rates
ERROR_RATE = Counter(
    'jaf_errors_total',
    'Total errors',
    ['agent_name', 'error_type']
)
```

#### Resource Metrics
```python
# Memory usage
MEMORY_USAGE = Gauge(
    'jaf_memory_usage_bytes',
    'Memory usage by component',
    ['component', 'agent_name']
)

# CPU usage
CPU_USAGE = Gauge(
    'jaf_cpu_usage_percent',
    'CPU usage by component',
    ['component']
)

# Active connections
ACTIVE_CONNECTIONS = Gauge(
    'jaf_active_connections',
    'Number of active connections',
    ['connection_type']
)
```

#### Business Metrics
```python
# Conversation metrics
CONVERSATION_LENGTH = Histogram(
    'jaf_conversation_length_messages',
    'Number of messages per conversation',
    buckets=[1, 5, 10, 20, 50, 100, float('inf')]
)

# Tool usage
TOOL_USAGE = Counter(
    'jaf_tool_usage_total',
    'Tool usage count',
    ['tool_name', 'agent_name']
)

# Model provider usage
MODEL_CALLS = Counter(
    'jaf_model_calls_total',
    'Model API calls',
    ['provider', 'model', 'status']
)
```

### Custom Metrics Collection

```python
class JAFMetricsCollector:
    def __init__(self):
        self.metrics = {
            'agent_requests': Counter('jaf_agent_requests_total', 'Total requests', ['agent', 'status']),
            'response_time': Histogram('jaf_response_time_seconds', 'Response time', ['agent']),
            'active_sessions': Gauge('jaf_active_sessions', 'Active sessions'),
            'memory_usage': Gauge('jaf_memory_usage_bytes', 'Memory usage', ['component']),
            'tool_executions': Counter('jaf_tool_executions_total', 'Tool executions', ['tool', 'status'])
        }
    
    def record_request(self, agent_name: str, duration: float, status: str):
        self.metrics['agent_requests'].labels(agent=agent_name, status=status).inc()
        self.metrics['response_time'].labels(agent=agent_name).observe(duration)
    
    def update_active_sessions(self, count: int):
        self.metrics['active_sessions'].set(count)
    
    def record_memory_usage(self, component: str, bytes_used: int):
        self.metrics['memory_usage'].labels(component=component).set(bytes_used)
    
    def record_tool_execution(self, tool_name: str, status: str):
        self.metrics['tool_executions'].labels(tool=tool_name, status=status).inc()

# Usage in your application
metrics = JAFMetricsCollector()

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status = 'success'
    except Exception as e:
        status = 'error'
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_request('api', duration, status)
    
    return response
```

## Logging Strategy

### Structured Logging

Use structured logging for better searchability and analysis:

```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage in agents
class LoggedAgent:
    def __init__(self, name):
        self.logger = logger.bind(agent_name=name)
    
    async def process_message(self, message, context):
        self.logger.info(
            "Processing message",
            message_id=message.id,
            context_id=context.id,
            user_id=context.user_id
        )
        
        try:
            result = await self._execute(message, context)
            
            self.logger.info(
                "Message processed successfully",
                message_id=message.id,
                response_length=len(result.content),
                processing_time=result.duration
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Message processing failed",
                message_id=message.id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise
```

### Log Levels and Categories

```python
# Different log levels for different scenarios
logger.debug("Detailed debug information", user_input=sanitized_input)
logger.info("Normal operation", session_id=session.id, action="message_sent")
logger.warning("Potential issue", warning_type="rate_limit_approaching", current_rate=95)
logger.error("Error occurred", error_code="AGENT_TIMEOUT", agent_name="MathTutor")
logger.critical("Critical system failure", component="memory_provider", error="connection_lost")

# Category-based logging
audit_logger = structlog.get_logger("audit")
security_logger = structlog.get_logger("security")
performance_logger = structlog.get_logger("performance")

# Audit logging
audit_logger.info(
    "User action",
    user_id=user.id,
    action="agent_query",
    agent_name="ChatBot",
    timestamp=datetime.utcnow().isoformat()
)

# Security logging
security_logger.warning(
    "Suspicious activity",
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent"),
    rate_limit_exceeded=True
)

# Performance logging
performance_logger.info(
    "Slow query detected",
    query_duration=5.2,
    agent_name="DatabaseAgent",
    query_type="complex_search"
)
```

### Log Aggregation

#### ELK Stack Configuration

**Logstash Configuration** (`logstash.conf`):
```ruby
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "jaf" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" or [level] == "CRITICAL" {
      mutate {
        add_tag => ["alert"]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "jaf-logs-%{+YYYY.MM.dd}"
  }
  
  if "alert" in [tags] {
    email {
      to => ["alerts@company.com"]
      subject => "JAF Alert: %{level} in %{agent_name}"
      body => "Error: %{message}\nTimestamp: %{timestamp}"
    }
  }
}
```

**Filebeat Configuration** (`filebeat.yml`):
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/*.json
  fields:
    service: jaf
    environment: production
  fields_under_root: true

output.logstash:
  hosts: ["logstash:5044"]

processors:
- add_host_metadata:
    when.not.contains.tags: forwarded
```

#### Fluentd Configuration

```ruby
<source>
  @type tail
  path /app/logs/*.json
  pos_file /var/log/fluentd/jaf.log.pos
  tag jaf.*
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%LZ
</source>

<filter jaf.**>
  @type record_transformer
  <record>
    service jaf
    environment "#{ENV['ENVIRONMENT']}"
    hostname "#{Socket.gethostname}"
  </record>
</filter>

<match jaf.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name jaf-logs
  type_name _doc
  include_tag_key true
  tag_key @log_name
  
  <buffer>
    flush_interval 10s
    chunk_limit_size 8m
    queue_limit_length 32
    retry_max_interval 30
    retry_forever true
  </buffer>
</match>
```

## Alerting

### Prometheus Alerting Rules

```yaml
# alerting-rules.yml
groups:
- name: jaf-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(jaf_agent_requests_total{status="error"}[5m]) / rate(jaf_agent_requests_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected for JAF agents"
      description: "Error rate is {{ $value | humanizePercentage }} for agent {{ $labels.agent_name }}"

  - alert: SlowResponseTime
    expr: histogram_quantile(0.95, rate(jaf_response_time_seconds_bucket[5m])) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow response times detected"
      description: "95th percentile response time is {{ $value }}s for agent {{ $labels.agent_name }}"

  - alert: HighMemoryUsage
    expr: jaf_memory_usage_bytes / (1024*1024*1024) > 2
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}GB for component {{ $labels.component }}"

  - alert: AgentDown
    expr: up{job="jaf-agents"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "JAF agent is down"
      description: "JAF agent {{ $labels.instance }} has been down for more than 1 minute"

  - alert: TooManyActiveConversations
    expr: jaf_active_conversations > 1000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High number of active conversations"
      description: "There are {{ $value }} active conversations, which may impact performance"
```

### Custom Alert Handlers

```python
import smtplib
import slack_sdk
from email.mime.text import MIMEText
from typing import Dict, Any

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slack_client = slack_sdk.WebClient(token=config.get('slack_token'))
    
    async def send_alert(self, alert_type: str, severity: str, message: str, context: Dict[str, Any] = None):
        """Send alert through multiple channels based on severity."""
        
        if severity in ['critical', 'high']:
            await self._send_slack_alert(alert_type, message, context)
            await self._send_email_alert(alert_type, message, context)
        elif severity == 'medium':
            await self._send_slack_alert(alert_type, message, context)
        else:
            # Log only for low severity
            logger.warning("Alert", type=alert_type, message=message, context=context)
    
    async def _send_slack_alert(self, alert_type: str, message: str, context: Dict[str, Any]):
        """Send alert to Slack."""
        try:
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 JAF Alert: {alert_type}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Message:* {message}"
                    }
                }
            ]
            
            if context:
                context_text = "\n".join([f"*{k}:* {v}" for k, v in context.items()])
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Context:*\n{context_text}"
                    }
                })
            
            await self.slack_client.chat_postMessage(
                channel=self.config['slack_channel'],
                blocks=blocks
            )
        except Exception as e:
            logger.error("Failed to send Slack alert", error=str(e))
    
    async def _send_email_alert(self, alert_type: str, message: str, context: Dict[str, Any]):
        """Send alert via email."""
        try:
            subject = f"JAF Alert: {alert_type}"
            body = f"Message: {message}\n\n"
            
            if context:
                body += "Context:\n"
                for key, value in context.items():
                    body += f"  {key}: {value}\n"
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.config['email_from']
            msg['To'] = ', '.join(self.config['email_to'])
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['smtp_user'], self.config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error("Failed to send email alert", error=str(e))

# Usage in monitoring
alert_manager = AlertManager(config['alerting'])

# Monitor error rates
async def check_error_rates():
    error_rate = await get_error_rate_last_5_minutes()
    if error_rate > 0.1:
        await alert_manager.send_alert(
            alert_type="HighErrorRate",
            severity="critical",
            message=f"Error rate is {error_rate:.2%}",
            context={"threshold": "10%", "current_rate": f"{error_rate:.2%}"}
        )
```

## Health Checks

### Comprehensive Health Monitoring

```python
from typing import Dict, List
import asyncio
import aiohttp
import time

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.last_results = {}
    
    def register_check(self, name: str, check_func, critical: bool = False):
        """Register a health check function."""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {},
            'summary': {
                'total': len(self.checks),
                'passed': 0,
                'failed': 0,
                'critical_failed': 0
            }
        }
        
        # Run all checks concurrently
        check_tasks = []
        for name, check_info in self.checks.items():
            task = asyncio.create_task(self._run_single_check(name, check_info))
            check_tasks.append(task)
        
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # Process results
        for i, (name, check_info) in enumerate(self.checks.items()):
            result = check_results[i]
            
            if isinstance(result, Exception):
                check_result = {
                    'status': 'failed',
                    'error': str(result),
                    'duration': 0,
                    'critical': check_info['critical']
                }
            else:
                check_result = result
                check_result['critical'] = check_info['critical']
            
            results['checks'][name] = check_result
            
            # Update summary
            if check_result['status'] == 'passed':
                results['summary']['passed'] += 1
            else:
                results['summary']['failed'] += 1
                if check_info['critical']:
                    results['summary']['critical_failed'] += 1
        
        # Determine overall status
        if results['summary']['critical_failed'] > 0:
            results['status'] = 'critical'
        elif results['summary']['failed'] > 0:
            results['status'] = 'degraded'
        
        self.last_results = results
        return results
    
    async def _run_single_check(self, name: str, check_info: Dict) -> Dict[str, Any]:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            result = await check_info['func']()
            duration = time.time() - start_time
            
            return {
                'status': 'passed' if result else 'failed',
                'duration': duration,
                'details': result if isinstance(result, dict) else {}
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'status': 'failed',
                'error': str(e),
                'duration': duration
            }

# Example health checks
async def check_database_connection():
    """Check database connectivity."""
    try:
        async with database.acquire() as conn:
            await conn.execute("SELECT 1")
        return {'connection': 'ok', 'pool_size': database.pool.size}
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")

async def check_redis_connection():
    """Check Redis connectivity."""
    try:
        await redis_client.ping()
        info = await redis_client.info()
        return {
            'connection': 'ok',
            'memory_used': info.get('used_memory_human'),
            'connected_clients': info.get('connected_clients')
        }
    except Exception as e:
        raise Exception(f"Redis connection failed: {e}")

async def check_model_provider():
    """Check model provider availability."""
    try:
        response = await model_provider.health_check()
        return {'provider': 'available', 'models': response.get('models', [])}
    except Exception as e:
        raise Exception(f"Model provider check failed: {e}")

async def check_memory_usage():
    """Check system memory usage."""
    import psutil
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        raise Exception(f"High memory usage: {memory.percent}%")
    return {
        'usage_percent': memory.percent,
        'available_mb': memory.available // 1024 // 1024
    }

# Register health checks
health_checker = HealthChecker()
health_checker.register_check('database', check_database_connection, critical=True)
health_checker.register_check('redis', check_redis_connection, critical=True)
health_checker.register_check('model_provider', check_model_provider, critical=False)
health_checker.register_check('memory', check_memory_usage, critical=False)

# Health endpoint
@app.get("/health")
async def health_endpoint():
    results = await health_checker.run_all_checks()
    
    status_code = 200
    if results['status'] == 'critical':
        status_code = 503
    elif results['status'] == 'degraded':
        status_code = 207
    
    return Response(content=json.dumps(results), status_code=status_code)
```

## Performance Monitoring

### Application Performance Monitoring (APM)

```python
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument your code
class TracedAgent:
    def __init__(self, name):
        self.name = name
        self.tracer = trace.get_tracer(f"agent.{name}")
    
    async def process_message(self, message, context):
        with self.tracer.start_as_current_span("process_message") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("message.id", message.id)
            span.set_attribute("context.user_id", context.user_id)
            
            try:
                # Process the message
                with self.tracer.start_as_current_span("extract_intent"):
                    intent = await self._extract_intent(message)
                    span.set_attribute("message.intent", intent)
                
                with self.tracer.start_as_current_span("generate_response"):
                    response = await self._generate_response(intent, context)
                    span.set_attribute("response.length", len(response))
                
                span.set_attribute("status", "success")
                return response
                
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("status", "error")
                raise
```

### Langfuse Tracing

JAF also supports tracing with [Langfuse](https://langfuse.com/). To enable it, set the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` environment variables.

```python
import os
from jaf.core.tracing import create_composite_trace_collector, ConsoleTraceCollector

# Set your Langfuse credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = "your_public_key"
os.environ["LANGFUSE_SECRET_KEY"] = "your_secret_key"

# The Langfuse collector will be added automatically
trace_collector = create_composite_trace_collector(ConsoleTraceCollector())

# Use this collector in your RunConfig
config = RunConfig(
    # ... other config
    on_event=trace_collector.collect,
)
```

See the full example at `examples/langfuse_tracing_demo.py`.

### Database Query Monitoring

```python
import asyncpg
import time
from contextlib import asynccontextmanager

class MonitoredDatabase:
    def __init__(self, pool):
        self.pool = pool
        self.slow_query_threshold = 1.0  # seconds
    
    @asynccontextmanager
    async def acquire(self):
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                # Wrap connection to monitor queries
                monitored_conn = MonitoredConnection(conn, self.slow_query_threshold)
                yield monitored_conn
        finally:
            duration = time.time() - start_time
            if duration > 5.0:  # Log slow connection acquisitions
                logger.warning("Slow connection acquisition", duration=duration)

class MonitoredConnection:
    def __init__(self, conn, slow_query_threshold):
        self.conn = conn
        self.slow_query_threshold = slow_query_threshold
    
    async def execute(self, query, *args):
        start_time = time.time()
        
        try:
            result = await self.conn.execute(query, *args)
            duration = time.time() - start_time
            
            # Log slow queries
            if duration > self.slow_query_threshold:
                logger.warning(
                    "Slow query detected",
                    query=query[:100],
                    duration=duration,
                    args_count=len(args)
                )
            
            # Record metrics
            DB_QUERY_DURATION.observe(duration)
            DB_QUERIES_TOTAL.labels(status='success').inc()
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            DB_QUERIES_TOTAL.labels(status='error').inc()
            
            logger.error(
                "Query failed",
                query=query[:100],
                duration=duration,
                error=str(e)
            )
            raise
```

## Dashboards

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "JAF Agent Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(jaf_agent_requests_total[5m])",
            "legendFormat": "{{agent_name}} - {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(jaf_response_time_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(jaf_response_time_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(jaf_agent_requests_total{status=\"error\"}[5m]) / rate(jaf_agent_requests_total[5m])",
            "format": "percent"
          }
        ]
      }
    ]
  }
}
```

### Custom Dashboard Queries

```promql
# Request rate by agent
rate(jaf_agent_requests_total[5m])

# Error rate percentage
(rate(jaf_agent_requests_total{status="error"}[5m]) / rate(jaf_agent_requests_total[5m])) * 100

# Response time percentiles
histogram_quantile(0.95, rate(jaf_response_time_seconds_bucket[5m]))

# Memory usage by component
jaf_memory_usage_bytes / (1024 * 1024 * 1024)

# Active conversations over time
jaf_active_conversations

# Tool usage frequency
rate(jaf_tool_executions_total[1h])

# Model API call success rate
rate(jaf_model_calls_total{status="success"}[5m]) / rate(jaf_model_calls_total[5m])
```

## Best Practices

### Monitoring Strategy

1. **Layer your monitoring**: Infrastructure → Application → Business metrics
2. **Monitor the user experience**: Response times, error rates, availability
3. **Set up proactive alerting**: Don't wait for users to report issues
4. **Use structured logging**: Makes searching and analysis much easier
5. **Monitor dependencies**: Database, Redis, model providers, external APIs
6. **Track business metrics**: Conversation success rates, user satisfaction

### Alert Management

1. **Avoid alert fatigue**: Only alert on actionable issues
2. **Use appropriate severity levels**: Critical, Warning, Info
3. **Provide context**: Include relevant information for troubleshooting
4. **Test your alerts**: Ensure they work when you need them
5. **Document runbooks**: What to do when each alert fires

### Performance Optimization

1. **Monitor resource usage**: CPU, memory, network, disk
2. **Track slow operations**: Database queries, API calls, model inference
3. **Use profiling**: Identify bottlenecks in your code
4. **Monitor external dependencies**: Third-party APIs and services
5. **Set up capacity planning**: Predict when you'll need to scale

This comprehensive monitoring setup ensures you have full visibility into your JAF applications and can maintain high reliability in production.