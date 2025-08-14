# Deployment Guide

This comprehensive guide covers deploying JAF applications to production environments using Docker, Kubernetes, and cloud platforms.

## Overview

JAF applications can be deployed in various configurations:

- **Development**: Local server with in-memory storage
- **Staging**: Docker containers with Redis/PostgreSQL
- **Production**: Kubernetes clusters with managed services
- **Serverless**: Cloud functions with external memory providers

## Docker Deployment

### Basic Dockerfile

Create a `Dockerfile` for your JAF application:

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Requirements File

Create `requirements.txt` for your JAF application:

```txt
# JAF Framework
jaf-py>=2.0.0

# Web server
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Model providers
litellm>=1.0.0
openai>=1.0.0

# Memory providers (optional)
redis>=5.0.0
asyncpg>=0.29.0

# Monitoring and logging
structlog>=23.0.0
prometheus-client>=0.19.0

# Environment and configuration
python-dotenv>=1.0.0
pydantic-settings>=2.0.0

# HTTP client
httpx>=0.25.0

# Development tools (optional)
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.1.0
```

### Build and Run

```bash
# Build the Docker image
docker build -t jaf-app:latest .

# Run the container
docker run -p 8000:8000 \
  -e LITELLM_URL=http://host.docker.internal:4000 \
  -e LITELLM_API_KEY=your-api-key \
  -e JAF_MEMORY_TYPE=memory \
  jaf-app:latest
```

### Multi-stage Build (Production)

For optimized production images:

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose Setup

### Basic Configuration

Create `docker-compose.yml` for local development:

```yaml
version: '3.8'

services:
  # JAF Application
  jaf-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LITELLM_URL=http://litellm:4000
      - LITELLM_API_KEY=${LITELLM_API_KEY}
      - JAF_MEMORY_TYPE=redis
      - JAF_REDIS_HOST=redis
      - JAF_REDIS_PORT=6379
      - JAF_REDIS_PASSWORD=${REDIS_PASSWORD}
    depends_on:
      - redis
      - litellm
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  # LiteLLM Proxy
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    environment:
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./litellm_config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000", "--num_workers", "1"]
    restart: unless-stopped

  # Redis for memory
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # PostgreSQL (alternative to Redis)
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=jaf_memory
      - POSTGRES_USER=jaf
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### Environment Configuration

Create `.env` file:

```bash
# LiteLLM Configuration
LITELLM_API_KEY=your-master-api-key
LITELLM_MASTER_KEY=your-master-key

# Model Provider API Keys
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-api-key

# Database Passwords
REDIS_PASSWORD=your-redis-password
POSTGRES_PASSWORD=your-postgres-password

# Monitoring
GRAFANA_PASSWORD=your-grafana-password
```

### LiteLLM Configuration

Create `litellm_config.yaml`:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-pro
      api_key: os.environ/GOOGLE_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: "postgresql://jaf:${POSTGRES_PASSWORD}@postgres:5432/jaf_memory"
  
  # Rate limiting
  rpm_limit: 1000
  tpm_limit: 100000

  # Caching
  redis_host: redis
  redis_port: 6379
  redis_password: os.environ/REDIS_PASSWORD

  # Logging
  set_verbose: true
  json_logs: true
```

### Database Initialization

Create `init.sql` for PostgreSQL:

```sql
-- JAF Memory Tables
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    messages JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at);

-- LiteLLM Tables (if needed)
CREATE DATABASE litellm_logs;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE jaf_memory TO jaf;
GRANT ALL PRIVILEGES ON DATABASE litellm_logs TO jaf;
```

### Production Docker Compose

Create `docker-compose.prod.yml` for production:

```yaml
version: '3.8'

services:
  jaf-app:
    image: your-registry/jaf-app:${APP_VERSION}
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LITELLM_URL=http://litellm:4000
      - LITELLM_API_KEY=${LITELLM_API_KEY}
      - JAF_MEMORY_TYPE=postgres
      - JAF_POSTGRES_HOST=postgres
      - JAF_POSTGRES_USERNAME=jaf
      - JAF_POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - JAF_POSTGRES_DATABASE=jaf_memory
    depends_on:
      - postgres
      - litellm
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    environment:
      - ENVIRONMENT=production
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}
      - DATABASE_URL=postgresql://jaf:${POSTGRES_PASSWORD}@postgres:5432/litellm_logs
    volumes:
      - ./litellm_config.yaml:/app/config.yaml:ro
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
    command: ["--config", "/app/config.yaml", "--port", "4000"]

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=jaf_memory
      - POSTGRES_USER=jaf
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    deploy:
      restart_policy:
        condition: on-failure
    command: postgres -c shared_preload_libraries=pg_stat_statements

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - jaf-app
    deploy:
      restart_policy:
        condition: on-failure

volumes:
  postgres_data:
    driver: local
```

## Kubernetes Deployment

### Namespace and ConfigMap

Create `k8s/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: jaf-system
  labels:
    name: jaf-system
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaf-config
  namespace: jaf-system
data:
  JAF_MEMORY_TYPE: "postgres"
  JAF_POSTGRES_HOST: "postgres-service"
  JAF_POSTGRES_DATABASE: "jaf_memory"
  JAF_POSTGRES_USERNAME: "jaf"
  LITELLM_URL: "http://litellm-service:4000"
  ENVIRONMENT: "production"
```

### Secrets

Create `k8s/secrets.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jaf-secrets
  namespace: jaf-system
type: Opaque
data:
  # Base64 encoded values
  LITELLM_API_KEY: eW91ci1hcGkta2V5  # your-api-key
  POSTGRES_PASSWORD: eW91ci1wb3N0Z3Jlcy1wYXNzd29yZA==  # your-postgres-password
  OPENAI_API_KEY: c2steW91ci1vcGVuYWkta2V5  # sk-your-openai-key
  ANTHROPIC_API_KEY: eW91ci1hbnRocm9waWMta2V5  # your-anthropic-key
```

### Application Deployment

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaf-app
  namespace: jaf-system
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
        image: your-registry/jaf-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: JAF_MEMORY_TYPE
          valueFrom:
            configMapKeyRef:
              name: jaf-config
              key: JAF_MEMORY_TYPE
        - name: JAF_POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: jaf-config
              key: JAF_POSTGRES_HOST
        - name: JAF_POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: jaf-secrets
              key: POSTGRES_PASSWORD
        - name: LITELLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: jaf-secrets
              key: LITELLM_API_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: jaf-app-service
  namespace: jaf-system
spec:
  selector:
    app: jaf-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### PostgreSQL StatefulSet

Create `k8s/postgres.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: jaf-system
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "jaf_memory"
        - name: POSTGRES_USER
          value: "jaf"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: jaf-secrets
              key: POSTGRES_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: jaf-system
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  type: ClusterIP
```

### Ingress Configuration

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: jaf-ingress
  namespace: jaf-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - jaf.yourdomain.com
    secretName: jaf-tls
  rules:
  - host: jaf.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: jaf-app-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

Create `k8s/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: jaf-app-hpa
  namespace: jaf-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: jaf-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Cloud Provider Deployments

### AWS Deployment

#### Using ECS with Fargate

Create `ecs-task-definition.json`:

```json
{
  "family": "jaf-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/jaf-task-role",
  "containerDefinitions": [
    {
      "name": "jaf-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/jaf-app:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "JAF_MEMORY_TYPE",
          "value": "postgres"
        },
        {
          "name": "JAF_POSTGRES_HOST",
          "value": "your-rds-endpoint.region.rds.amazonaws.com"
        }
      ],
      "secrets": [
        {
          "name": "JAF_POSTGRES_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:jaf/postgres-password"
        },
        {
          "name": "LITELLM_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:jaf/litellm-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/jaf-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### CloudFormation Template

Create `cloudformation-template.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'JAF Application Infrastructure'

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC ID for deployment
  
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnet IDs for ECS service

Resources:
  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: jaf-cluster
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: jaf-alb
      Type: application
      Scheme: internet-facing
      Subnets: !Ref SubnetIds
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # RDS PostgreSQL Instance
  PostgreSQLDB:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: jaf-postgres
      DBInstanceClass: db.t3.micro
      Engine: postgres
      EngineVersion: '15.4'
      AllocatedStorage: 20
      MasterUsername: jaf
      MasterUserPassword: !Ref DBPassword
      VPCSecurityGroups:
        - !Ref RDSSecurityGroup
      DBSubnetGroupName: !Ref DBSubnetGroup

  # ElastiCache Redis
  RedisCluster:
    Type: AWS::ElastiCache::CacheCluster
    Properties:
      CacheNodeType: cache.t3.micro
      Engine: redis
      NumCacheNodes: 1
      VpcSecurityGroupIds:
        - !Ref RedisSecurityGroup

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: jaf-service
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 3
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          SecurityGroups:
            - !Ref ECSSecurityGroup
          Subnets: !Ref SubnetIds
          AssignPublicIp: ENABLED
      LoadBalancers:
        - ContainerName: jaf-app
          ContainerPort: 8000
          TargetGroupArn: !Ref TargetGroup

Outputs:
  LoadBalancerDNS:
    Description: DNS name of the load balancer
    Value: !GetAtt LoadBalancer.DNSName
    Export:
      Name: !Sub ${AWS::StackName}-LoadBalancerDNS
```

### Google Cloud Platform

#### Using Cloud Run

Create `cloudbuild.yaml`:

```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/jaf-app:$COMMIT_SHA', '.']

  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/jaf-app:$COMMIT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'jaf-app'
      - '--image=gcr.io/$PROJECT_ID/jaf-app:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--set-env-vars=JAF_MEMORY_TYPE=postgres'
      - '--set-env-vars=JAF_POSTGRES_HOST=${_POSTGRES_HOST}'
      - '--set-secrets=JAF_POSTGRES_PASSWORD=postgres-password:latest'
      - '--set-secrets=LITELLM_API_KEY=litellm-api-key:latest'
      - '--memory=2Gi'
      - '--cpu=2'
      - '--max-instances=10'
      - '--concurrency=100'

substitutions:
  _POSTGRES_HOST: 'your-postgres-instance-ip'

options:
  machineType: 'E2_HIGHCPU_8'
```

### Azure Deployment

#### Using Container Instances

Create `azure-container-group.yaml`:

```yaml
apiVersion: 2019-12-01
location: eastus
name: jaf-container-group
properties:
  containers:
  - name: jaf-app
    properties:
      image: your-registry.azurecr.io/jaf-app:latest
      resources:
        requests:
          cpu: 1.0
          memoryInGb: 2.0
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: JAF_MEMORY_TYPE
        value: postgres
      - name: JAF_POSTGRES_HOST
        value: your-postgres-server.postgres.database.azure.com
      - name: JAF_POSTGRES_PASSWORD
        secureValue: your-postgres-password
      - name: LITELLM_API_KEY
        secureValue: your-litellm-api-key
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8000
    dnsNameLabel: jaf-app-unique-label
tags:
  environment: production
  application: jaf
```

## Environment Configuration

### Production Environment Variables

Create comprehensive environment configuration:

```bash
# Application Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Provider Configuration
LITELLM_URL=https://your-litellm-proxy.com
LITELLM_API_KEY=your-secure-api-key
LITELLM_MODEL=gpt-4

# Memory Configuration
JAF_MEMORY_TYPE=postgres
JAF_POSTGRES_HOST=your-postgres-host.com
JAF_POSTGRES_PORT=5432
JAF_POSTGRES_DATABASE=jaf_memory
JAF_POSTGRES_USERNAME=jaf_user
JAF_POSTGRES_PASSWORD=your-secure-password
JAF_POSTGRES_SSL=true
JAF_POSTGRES_MAX_CONNECTIONS=20

# Security Configuration
CORS_ORIGINS=https://your-frontend.com,https://admin.your-frontend.com
API_RATE_LIMIT=100
API_RATE_WINDOW=60

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn.com
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# Caching Configuration
REDIS_URL=redis://your-redis-cluster.com:6379
REDIS_PASSWORD=your-redis-password
CACHE_TTL=3600

# External Services
WEBHOOK_URL=https://your-webhook-endpoint.com
EXTERNAL_API_KEY=your-external-api-key
```

### Configuration Management

Use proper configuration management:

```python
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    
    # Model Provider
    litellm_url: str = "http://localhost:4000"
    litellm_api_key: str
    litellm_model: str = "gpt-3.5-turbo"
    
    # Memory
    jaf_memory_type: str = "memory"
    jaf_postgres_host: Optional[str] = None
    jaf_postgres_password: Optional[str] = None
    
    # Security
    cors_origins: List[str] = ["*"]
    api_rate_limit: int = 100
    api_rate_window: int = 60
    
    # Monitoring
    enable_metrics: bool = False
    sentry_dsn: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
settings = Settings()
```

## Monitoring and Observability

### Prometheus Metrics

Add metrics to your JAF application:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('jaf_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('jaf_request_duration_seconds', 'Request duration')
ACTIVE_CONVERSATIONS = Gauge('jaf_active_conversations', 'Active conversations')

# Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# Start metrics server
if settings.enable_metrics:
    start_http_server(settings.metrics_port)
```

### Structured Logging

Configure structured logging:

```python
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage in application
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    logger.info("Request started", 
                method=request.method, 
                path=request.url.path,
                client_host=request.client.host)
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info("Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration)
    
    return response
```

### Health Checks

Implement comprehensive health checks:

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "checks": {}
    }
    
    # Check database connection
    try:
        if memory_provider:
            db_health = await memory_provider.health_check()
            health_data["checks"]["database"] = {
                "status": "healthy" if db_health.get("healthy") else "unhealthy",
                "latency_ms": db_health.get("latency_ms", 0)
            }
    except Exception as e:
        health_data["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check model provider
    try:
        # Simple test request
        test_response = await model_provider.get_completion(test_state, test_agent, test_config)
        health_data["checks"]["model_provider"] = {"status": "healthy"}
    except Exception as e:
        health_data["checks"]["model_provider"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Determine overall status
    all_healthy = all(
        check.get("status") == "healthy" 
        for check in health_data["checks"].values()
    )
    
    if not all_healthy:
        health_data["status"] = "unhealthy"
        return JSONResponse(content=health_data, status_code=503)
    
    return health_data
```

## Security Best Practices

### Container Security

```dockerfile
# Use specific version tags
FROM python:3.11.6-slim

# Don't run as root
RUN groupadd -r jaf && useradd -r -g jaf jaf

# Set proper file permissions
COPY --chown=jaf:jaf . /app
USER jaf

# Use read-only root filesystem
# Add to docker run: --read-only --tmpfs /tmp

# Limit capabilities
# Add to docker run: --cap-drop=ALL --cap-add=NET_BIND_SERVICE
```

### Secrets Management

```python
# Use external secret management
import boto3
from azure.keyvault.secrets import SecretClient

class SecretManager:
    def __init__(self, provider="aws"):
        self.provider = provider
        if provider == "aws":
            self.client = boto3.client('secretsmanager')
        elif provider == "azure":
            self.client = SecretClient(vault_url, credential)
    
    async def get_secret(self, secret_name: str) -> str:
        if self.provider == "aws":
            response = self.client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        elif self.provider == "azure":
            secret = self.client.get_secret(secret_name)
            return secret.value

# Usage
secret_manager = SecretManager()
api_key = await secret_manager.get_secret("jaf/litellm-api-key")
```

### Network Security

```yaml
# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: jaf-network-policy
  namespace: jaf-system
spec:
  podSelector:
    matchLabels:
      app: jaf-app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: jaf-system
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
  - to: []  # Allow external API calls
    ports:
    - protocol: TCP
      port: 443
```

## Troubleshooting Deployment Issues

### Common Issues

1. **Container Won't Start**
```bash
# Check container logs
docker logs container-id

# Check resource limits
docker stats container-id

# Verify environment variables
docker exec container-id env
```

2. **Database Connection Issues**
```bash
# Test database connectivity
docker exec -it container-id psql -h postgres-host -U username -d database

# Check security groups/firewall rules
telnet postgres-host 5432
```

3. **Memory Issues**
```bash
# Monitor memory usage
kubectl top pods -n jaf-system

# Check resource requests/limits
kubectl describe pod pod-name -n jaf-system
```

4. **Performance Issues**
```bash
# Check application metrics
curl http://localhost:9090/metrics

# Profile application
docker exec -it container-id python -m cProfile main.py
```

## Next Steps

- Review [Troubleshooting](troubleshooting.md) for common deployment issues
- Check [Monitoring](monitoring.md) for production observability
- Explore [Security](security.md) for hardening guidelines
- See [Examples](examples.md) for deployment configurations