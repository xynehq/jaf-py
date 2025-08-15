# Infrastructure

!!! info "Production Infrastructure"
    JAF provides production-ready infrastructure components including database providers, LLM integrations, and configuration management for enterprise deployment.

##  Overview

JAF's infrastructure layer provides:

- **💾 Database Providers**: Redis, PostgreSQL, and in-memory session storage
- **🤖 LLM Integrations**: Multi-provider support with real streaming
- ** Configuration Management**: Environment-based configuration
- **🔄 Service Discovery**: Automatic provider detection and health checking
- ** Monitoring**: Built-in metrics and observability

## 💾 Database Providers

### Redis Provider

```python
from adk.sessions import create_redis_session_provider

# Redis configuration
redis_config = {
    "url": "redis://localhost:6379",
    "max_connections": 20,
    "key_prefix": "jaf:session:",
    "ttl_seconds": 3600,
    "retry_attempts": 3
}

redis_provider = create_redis_session_provider(redis_config)

# Store and retrieve sessions
await redis_provider.store_session(session)
retrieved_session = await redis_provider.get_session(session_id)
```

### PostgreSQL Provider

```python
from adk.sessions import create_postgres_session_provider

# PostgreSQL configuration
postgres_config = {
    "url": "postgresql://user:pass@localhost:5432/jaf_db",
    "pool_size": 10,
    "max_overflow": 20,
    "table_name": "agent_sessions",
    "auto_create_tables": True
}

postgres_provider = create_postgres_session_provider(postgres_config)
```

### In-Memory Provider

```python
from adk.sessions import create_in_memory_session_provider

# In-memory configuration (development/testing)
memory_config = {
    "max_sessions": 10000,
    "ttl_seconds": 1800,
    "cleanup_interval": 300
}

memory_provider = create_in_memory_session_provider(memory_config)
```

## 🤖 LLM Service Integration

### Multi-Provider Support

```python
from adk.llm import (
    create_openai_llm_service,
    create_anthropic_llm_service,
    create_litellm_service
)

# OpenAI integration
openai_service = create_openai_llm_service({
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4",
    "timeout": 30,
    "max_retries": 3
})

# Anthropic integration  
anthropic_service = create_anthropic_llm_service({
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 4096
})

# LiteLLM proxy integration
litellm_service = create_litellm_service({
    "base_url": "http://localhost:4000",
    "api_key": "proxy-key",
    "model": "gpt-4"
})
```

### Real Streaming Implementation

```python
from adk.llm import StreamingLLMService

async def stream_llm_response(prompt: str):
    """Stream LLM response with real-time processing."""
    
    async for chunk in openai_service.stream_completion(
        prompt=prompt,
        stream_options={"include_usage": True}
    ):
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            
        if chunk.usage:
            # Final chunk with usage statistics
            print(f"Tokens used: {chunk.usage.total_tokens}")

# Usage
async for text_chunk in stream_llm_response("Explain quantum computing"):
    print(text_chunk, end="", flush=True)
```

##  Configuration Management

### Environment-Based Configuration

```python
from adk.config import AdkConfig, load_config_from_env

# Load configuration from environment variables
config = load_config_from_env()

# Manual configuration
config = AdkConfig(
    # Database settings
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    postgres_url=os.getenv("POSTGRES_URL"),
    
    # LLM settings
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    litellm_base_url=os.getenv("LITELLM_URL", "http://localhost:4000"),
    
    # Security settings
    security_level=os.getenv("ADK_SECURITY_LEVEL", "high"),
    enable_auth=os.getenv("ADK_ENABLE_AUTH", "true").lower() == "true",
    
    # Performance settings
    max_concurrent_sessions=int(os.getenv("ADK_MAX_SESSIONS", "1000")),
    session_timeout=int(os.getenv("ADK_SESSION_TIMEOUT", "3600"))
)
```

### Configuration Validation

```python
from adk.config import validate_config, ConfigValidationError

try:
    validation_result = validate_config(config)
    if not validation_result.is_valid:
        print(f"Configuration errors: {validation_result.errors}")
        sys.exit(1)
except ConfigValidationError as e:
    print(f"Invalid configuration: {e}")
    sys.exit(1)
```

## 🔄 Service Discovery and Health Checking

### Automatic Provider Detection

```python
from adk.infrastructure import ServiceDiscovery

service_discovery = ServiceDiscovery()

# Automatically discover available services
available_services = await service_discovery.discover_services([
    "redis://localhost:6379",
    "postgresql://localhost:5432/jaf_db",
    "http://localhost:4000"  # LiteLLM proxy
])

print(f"Available services: {[s.name for s in available_services]}")
```

### Health Monitoring

```python
from adk.infrastructure import HealthChecker

health_checker = HealthChecker()

# Register services for health checking
health_checker.register_service("redis", redis_provider)
health_checker.register_service("postgres", postgres_provider)
health_checker.register_service("llm", openai_service)

# Check overall system health
health_status = await health_checker.check_all_services()
print(f"System health: {health_status.overall_status}")

for service_name, status in health_status.service_statuses.items():
    print(f"{service_name}: {status.status} ({status.latency_ms}ms)")
```

##  Monitoring and Observability

### Metrics Collection

```python
from adk.infrastructure import MetricsCollector

metrics = MetricsCollector()

# Track infrastructure metrics
metrics.increment_counter("session.created")
metrics.record_histogram("llm.response_time", response_time_ms)
metrics.set_gauge("active_sessions", session_count)

# Custom metrics
metrics.track_custom_metric("business_metric", value, tags={
    "user_type": "premium",
    "feature": "advanced_agent"
})
```

### Distributed Tracing

```python
from adk.infrastructure import TracingConfig, setup_tracing

# Configure distributed tracing
tracing_config = TracingConfig(
    service_name="jaf-agent-system",
    jaeger_endpoint="http://localhost:14268",
    sample_rate=0.1  # Sample 10% of traces
)

setup_tracing(tracing_config)

# Automatic tracing for operations
@trace_operation("llm_call")
async def traced_llm_call(prompt: str):
    """LLM call with automatic tracing."""
    return await llm_service.complete(prompt)
```

##  Container Deployment

### Docker Configuration

```dockerfile
# Dockerfile for JAF application
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install JAF with production dependencies
RUN pip install "jaf-py[all]"

# Copy application code
COPY . .

# Set environment variables
ENV ADK_SECURITY_LEVEL=high
ENV ADK_ENABLE_AUTH=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import asyncio; from adk.infrastructure import health_check; asyncio.run(health_check())"

# Run application
CMD ["python", "-m", "adk.server"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  jaf-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/jaf_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ADK_SECURITY_LEVEL=high
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=jaf_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

## ☸️ Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaf-app
  labels:
    app: jaf-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jaf-app
  template:
    metadata:
      labels:
        app: jaf-app
    spec:
      containers:
      - name: jaf-app
        image: jaf-py:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: jaf-app-service
spec:
  selector:
    app: jaf-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

##  Security Infrastructure

### TLS/SSL Configuration

```python
from adk.infrastructure import SSLConfig, setup_ssl

# Configure SSL/TLS
ssl_config = SSLConfig(
    cert_file="/etc/ssl/certs/jaf-app.crt",
    key_file="/etc/ssl/private/jaf-app.key",
    ca_file="/etc/ssl/certs/ca-bundle.crt",
    verify_mode="required"
)

# Apply SSL configuration
setup_ssl(ssl_config)
```

### Network Security

```python
from adk.infrastructure import NetworkSecurityConfig

network_config = NetworkSecurityConfig(
    allowed_origins=["https://your-domain.com"],
    rate_limit_requests_per_minute=100,
    enable_cors=True,
    cors_max_age=86400,
    trusted_proxies=["10.0.0.0/8", "172.16.0.0/12"]
)
```

##  Performance Optimization

### Connection Pooling

```python
from adk.infrastructure import ConnectionPoolConfig

# Database connection pooling
db_pool_config = ConnectionPoolConfig(
    min_connections=5,
    max_connections=20,
    connection_timeout=30,
    idle_timeout=600,
    max_lifetime=3600
)

# LLM service pooling
llm_pool_config = ConnectionPoolConfig(
    min_connections=2,
    max_connections=10,
    connection_timeout=10,
    request_timeout=30
)
```

### Caching Strategy

```python
from adk.infrastructure import CacheConfig, setup_caching

cache_config = CacheConfig(
    backend="redis",
    default_ttl=300,  # 5 minutes
    max_entries=10000,
    cache_strategies={
        "llm_responses": {"ttl": 1800, "max_size": 1000},
        "session_data": {"ttl": 3600, "max_size": 5000},
        "user_preferences": {"ttl": 86400, "max_size": 10000}
    }
)

setup_caching(cache_config)
```

##  Infrastructure Management

### Automated Deployment

```python
from adk.infrastructure import DeploymentManager

deployment_manager = DeploymentManager()

# Deploy with zero-downtime
await deployment_manager.deploy({
    "strategy": "rolling_update",
    "max_unavailable": "25%",
    "max_surge": "25%",
    "health_check_grace_period": 60,
    "rollback_on_failure": True
})
```

### Backup and Recovery

```python
from adk.infrastructure import BackupManager

backup_manager = BackupManager()

# Automated backup configuration
await backup_manager.setup_automated_backups({
    "schedule": "0 2 * * *",  # Daily at 2 AM
    "retention_days": 30,
    "encrypt": True,
    "compression": True,
    "storage_backend": "s3"
})

# Manual backup
backup_id = await backup_manager.create_backup("manual-backup")
```

## 🔗 Related Documentation

- **[Deployment](deployment.md)** - Detailed deployment procedures
- **[Security Framework](security-framework.md)** - Security infrastructure
- **[Error Handling](error-handling.md)** - Infrastructure resilience
- **[Validation Suite](validation-suite.md)** - Infrastructure testing

---

!!! success "Production Infrastructure"
    JAF's infrastructure layer provides enterprise-grade components for database management, LLM integration, and service monitoring. All components are designed for scalability, reliability, and production deployment.