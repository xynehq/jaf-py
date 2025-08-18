# A2A Deployment Guide

Production deployment patterns and best practices for JAF Agent-to-Agent (A2A) servers.

## Overview

This guide covers deploying A2A servers in production environments, including containerization, load balancing, monitoring, and security considerations.

## Deployment Architecture

### Single Server Deployment

```
[Client] → [Load Balancer] → [A2A Server] → [Agents]
                                    ↓
                           [Memory Provider]
```

### Multi-Agent Distributed Deployment

```
[Client] → [API Gateway] → [Agent Router] → [Specialized Agents]
                              ↓               ↓
                         [Service Mesh]  [Agent Pool]
                              ↓               ↓
                        [Shared Memory] [Local Memory]
```

## Environment Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# Server Configuration
A2A_HOST=0.0.0.0
A2A_PORT=3000
A2A_CORS_ENABLED=true
A2A_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Authentication
A2A_AUTH_ENABLED=true
A2A_JWT_SECRET=your-jwt-secret-key
A2A_API_KEYS=key1,key2,key3

# Model Provider
LITELLM_URL=http://litellm-proxy:4000
LITELLM_API_KEY=your-litellm-api-key
LITELLM_MODEL=gpt-4

# Memory Provider
MEMORY_PROVIDER=redis
REDIS_URL=redis://redis-cluster:6379
REDIS_PASSWORD=your-redis-password

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
TRACE_ENABLED=true

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Configuration Validation

```python
import os
from jaf.a2a import validate_server_config

def load_config():
    """Load and validate server configuration."""
    config = {
        'host': os.getenv('A2A_HOST', '0.0.0.0'),
        'port': int(os.getenv('A2A_PORT', '3000')),
        'cors_enabled': os.getenv('A2A_CORS_ENABLED', 'false').lower() == 'true',
        'auth_enabled': os.getenv('A2A_AUTH_ENABLED', 'false').lower() == 'true',
        'memory_provider': os.getenv('MEMORY_PROVIDER', 'memory'),
        'rate_limit_enabled': os.getenv('RATE_LIMIT_ENABLED', 'false').lower() == 'true'
    }
    
    # Validate configuration
    errors = validate_server_config(config)
    if errors:
        raise ValueError(f"Configuration errors: {errors}")
    
    return config
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:3000/a2a/health || exit 1

# Start server
CMD ["python", "-m", "jaf.a2a.server"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  a2a-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - A2A_HOST=0.0.0.0
      - A2A_PORT=3000
      - MEMORY_PROVIDER=redis
      - REDIS_URL=redis://redis:6379
      - LITELLM_URL=http://litellm-proxy:4000
    depends_on:
      - redis
      - litellm-proxy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/a2a/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  litellm-proxy:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./litellm_config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - a2a-server
    restart: unless-stopped

volumes:
  redis_data:
```

### Multi-Stage Production Build

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:3000/a2a/health || exit 1

CMD ["python", "-m", "jaf.a2a.server"]
```

## Kubernetes Deployment

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: a2a-config
data:
  A2A_HOST: "0.0.0.0"
  A2A_PORT: "3000"
  A2A_CORS_ENABLED: "true"
  MEMORY_PROVIDER: "redis"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: a2a-secrets
type: Opaque
stringData:
  REDIS_PASSWORD: "your-redis-password"
  LITELLM_API_KEY: "your-litellm-api-key"
  JWT_SECRET: "your-jwt-secret"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a2a-server
  labels:
    app: a2a-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: a2a-server
  template:
    metadata:
      labels:
        app: a2a-server
    spec:
      containers:
      - name: a2a-server
        image: your-registry/a2a-server:latest
        ports:
        - containerPort: 3000
        - containerPort: 9090  # Metrics
        envFrom:
        - configMapRef:
            name: a2a-config
        - secretRef:
            name: a2a-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /a2a/health
            port: 3000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /a2a/health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: a2a-service
  labels:
    app: a2a-server
spec:
  selector:
    app: a2a-server
  ports:
  - name: http
    port: 80
    targetPort: 3000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: a2a-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: a2a-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /a2a
        pathType: Prefix
        backend:
          service:
            name: a2a-service
            port:
              number: 80
```

## Load Balancing

### Nginx Configuration

```nginx
upstream a2a_backend {
    least_conn;
    server a2a-server-1:3000 max_fails=3 fail_timeout=30s;
    server a2a-server-2:3000 max_fails=3 fail_timeout=30s;
    server a2a-server-3:3000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.example.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/api.example.com.crt;
    ssl_certificate_key /etc/nginx/ssl/api.example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=a2a:10m rate=10r/s;
    limit_req zone=a2a burst=20 nodelay;

    location /a2a {
        proxy_pass http://a2a_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # HTTP/1.1 support for SSE streaming
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }

    location /health {
        access_log off;
        proxy_pass http://a2a_backend/a2a/health;
    }
}
```

### HAProxy Configuration

```
global
    daemon
    maxconn 4096
    log stdout local0

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull

frontend a2a_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/api.example.com.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    default_backend a2a_backend

backend a2a_backend
    balance roundrobin
    option httpchk GET /a2a/health
    
    server a2a-1 a2a-server-1:3000 check inter 5s
    server a2a-2 a2a-server-2:3000 check inter 5s
    server a2a-3 a2a-server-3:3000 check inter 5s
```

## Security

### Authentication Setup

```python
from jaf.a2a import create_a2a_server, AuthConfig

auth_config = AuthConfig(
    enabled=True,
    jwt_secret=os.getenv('JWT_SECRET'),
    api_keys=os.getenv('API_KEYS', '').split(','),
    rate_limit={
        'requests_per_minute': 100,
        'burst_size': 20
    }
)

server = create_a2a_server(
    agents=agents,
    auth_config=auth_config,
    cors_config={
        'enabled': True,
        'origins': ['https://app.example.com'],
        'credentials': True
    }
)
```

### TLS Configuration

```python
import ssl

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('/path/to/cert.pem', '/path/to/key.pem')

# Run with HTTPS
await server.start(
    host='0.0.0.0',
    port=443,
    ssl=ssl_context
)
```

### Network Security

```yaml
# Network Policy (Kubernetes)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: a2a-network-policy
spec:
  podSelector:
    matchLabels:
      app: a2a-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 6379  # Redis
```

## Monitoring and Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server

# Custom metrics
REQUEST_COUNT = Counter('a2a_requests_total', 'Total A2A requests', ['method', 'status'])
REQUEST_DURATION = Histogram('a2a_request_duration_seconds', 'A2A request duration')

class MetricsMiddleware:
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(method=request.method, status=response.status_code).inc()
            return response
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)

# Start metrics server
start_http_server(9090)
```

### Logging Configuration

```python
import logging
import structlog

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
```

### Health Checks

```python
async def detailed_health_check():
    """Comprehensive health check."""
    checks = {
        'server': 'healthy',
        'agents': {},
        'memory': 'unknown',
        'model_provider': 'unknown'
    }
    
    # Check agents
    for name, agent in agents.items():
        try:
            await agent.health_check()
            checks['agents'][name] = 'healthy'
        except Exception as e:
            checks['agents'][name] = f'unhealthy: {e}'
    
    # Check memory provider
    try:
        await memory_provider.health_check()
        checks['memory'] = 'healthy'
    except Exception as e:
        checks['memory'] = f'unhealthy: {e}'
    
    # Check model provider
    try:
        await model_provider.health_check()
        checks['model_provider'] = 'healthy'
    except Exception as e:
        checks['model_provider'] = f'unhealthy: {e}'
    
    return checks
```

## Performance Optimization

### Connection Pooling

```python
import asyncio
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine

class ConnectionManager:
    def __init__(self):
        self.redis_pool = None
        self.db_engine = None
    
    async def initialize(self):
        # Redis connection pool
        self.redis_pool = aioredis.ConnectionPool.from_url(
            "redis://localhost:6379",
            max_connections=20,
            retry_on_timeout=True
        )
        
        # Database connection pool
        self.db_engine = create_async_engine(
            "postgresql+asyncpg://user:pass@localhost/db",
            pool_size=20,
            max_overflow=0,
            pool_pre_ping=True
        )
    
    async def close(self):
        if self.redis_pool:
            await self.redis_pool.disconnect()
        if self.db_engine:
            await self.db_engine.dispose()
```

### Caching Strategy

```python
from functools import wraps
import json
import hashlib

def cache_response(ttl=300):
    """Cache agent responses."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = hashlib.md5(
                json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            cached = await redis.get(f"response:{cache_key}")
            if cached:
                return json.loads(cached)
            
            # Generate response
            response = await func(*args, **kwargs)
            
            # Cache response
            await redis.setex(f"response:{cache_key}", ttl, json.dumps(response))
            
            return response
        return wrapper
    return decorator
```

## Troubleshooting

### Common Issues

1. **Memory Leaks**
   ```python
   # Monitor memory usage
   import psutil
   
   process = psutil.Process()
   memory_mb = process.memory_info().rss / 1024 / 1024
   if memory_mb > 500:  # Alert threshold
       logger.warning("High memory usage", memory_mb=memory_mb)
   ```

2. **Connection Pool Exhaustion**
   ```python
   # Monitor connection pools
   if redis_pool.created_connections > redis_pool.max_connections * 0.9:
       logger.warning("Redis pool nearly exhausted")
   ```

3. **Rate Limiting Issues**
   ```python
   # Implement backoff strategy
   import asyncio
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   async def send_with_backoff(client, message):
       return await client.send_message(message)
   ```

### Debug Mode

```python
# Enable debug mode
if os.getenv('DEBUG', 'false').lower() == 'true':
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Add request/response logging
    @app.middleware("http")
    async def log_requests(request, call_next):
        logger.debug("Request", method=request.method, url=str(request.url))
        response = await call_next(request)
        logger.debug("Response", status=response.status_code)
        return response
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# Backup Redis
redis-cli --rdb /backup/redis_${DATE}.rdb

# Backup PostgreSQL
pg_dump -h postgres-host -U username -d database > /backup/postgres_${DATE}.sql

# Backup configuration
cp -r /app/config /backup/config_${DATE}/

# Upload to S3
aws s3 sync /backup/ s3://your-backup-bucket/a2a-backups/
```

### Disaster Recovery

```yaml
# disaster-recovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dr-procedures
data:
  recovery.sh: |
    #!/bin/bash
    echo "Starting disaster recovery..."
    
    # Restore Redis from backup
    redis-cli --rdb /backup/redis_latest.rdb
    
    # Restore PostgreSQL
    psql -h postgres-host -U username -d database < /backup/postgres_latest.sql
    
    # Restart services
    kubectl rollout restart deployment/a2a-server
    
    echo "Recovery complete"
```

## Related Documentation

- [A2A Protocol Overview](a2a-protocol.md)
- [A2A API Reference](a2a-api-reference.md)
- [Monitoring Guide](monitoring.md)
- [Security Guide](security.md)