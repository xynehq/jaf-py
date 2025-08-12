# Model Providers

JAF integrates with Large Language Models (LLMs) through a flexible provider system. The primary provider is LiteLLM, which offers unified access to multiple LLM services including OpenAI, Anthropic, Google, and local models.

## Overview

Model providers in JAF handle the communication between your agents and LLM services. They:

- Convert JAF messages to provider-specific formats
- Handle tool calling and function execution
- Manage model configuration and parameters
- Provide a consistent interface across different LLM providers

## LiteLLM Provider

LiteLLM is the recommended and primary model provider for JAF. It acts as a proxy that translates requests to different LLM APIs using a unified interface.

### Basic Setup

```python
from jaf.providers.model import make_litellm_provider

# Create provider instance
provider = make_litellm_provider(
    base_url="http://localhost:4000",  # LiteLLM server URL
    api_key="your-api-key"             # API key (optional for local servers)
)

# Use with JAF
config = RunConfig(
    agent_registry={"MyAgent": my_agent},
    model_provider=provider,
    model_override="gpt-4"  # Optional: override model
)
```

### LiteLLM Server Setup

LiteLLM can run as a server that proxies requests to various LLM providers:

```bash
# Install LiteLLM
pip install litellm[proxy]

# Start LiteLLM server
litellm --config config.yaml --port 4000
```

**LiteLLM Configuration Example (`config.yaml`):**

```yaml
model_list:
  # OpenAI models
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-your-openai-key
      
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: sk-your-openai-key

  # Anthropic models  
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: your-anthropic-key
      
  # Google models
  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-pro
      api_key: your-google-api-key
      
  # Local models via Ollama
  - model_name: llama2
    litellm_params:
      model: ollama/llama2
      api_base: http://localhost:11434

  # Azure OpenAI
  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4
      api_key: your-azure-key
      api_base: https://your-resource.openai.azure.com/
      api_version: "2023-07-01-preview"

general_settings:
  master_key: your-master-key  # For authentication
  database_url: "postgresql://user:pass@localhost/litellm"  # Optional: for logging
```

## Supported LLM Providers

### 1. OpenAI

```python
# Direct OpenAI configuration in LiteLLM
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-your-openai-api-key
      organization: your-org-id  # Optional

# Environment variables
export OPENAI_API_KEY=sk-your-openai-api-key
export OPENAI_ORGANIZATION=your-org-id
```

**Supported Models:**
- `gpt-4`, `gpt-4-turbo`, `gpt-4o`
- `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
- `text-davinci-003`, `text-curie-001`

### 2. Anthropic Claude

```python
# Anthropic configuration
model_list:
  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: your-anthropic-api-key

# Environment variables
export ANTHROPIC_API_KEY=your-anthropic-api-key
```

**Supported Models:**
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-2.1`, `claude-2.0`
- `claude-instant-1.2`

### 3. Google (Gemini/PaLM)

```python
# Google configuration
model_list:
  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-pro
      api_key: your-google-api-key

# Environment variables
export GOOGLE_API_KEY=your-google-api-key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

**Supported Models:**
- `gemini-pro`, `gemini-pro-vision`
- `gemini-1.5-pro`, `gemini-1.5-flash`
- `text-bison-001`, `chat-bison-001`

### 4. Local Models (Ollama)

```python
# Ollama configuration
model_list:
  - model_name: llama2
    litellm_params:
      model: ollama/llama2
      api_base: http://localhost:11434
      
  - model_name: mistral
    litellm_params:
      model: ollama/mistral
      api_base: http://localhost:11434
```

**Setup Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download and run models
ollama pull llama2
ollama pull mistral
ollama pull codellama

# Start Ollama server (if not auto-started)
ollama serve
```

### 5. Azure OpenAI

```python
# Azure OpenAI configuration
model_list:
  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4
      api_key: your-azure-api-key
      api_base: https://your-resource.openai.azure.com/
      api_version: "2023-07-01-preview"
      
# Environment variables
export AZURE_API_KEY=your-azure-api-key
export AZURE_API_BASE=https://your-resource.openai.azure.com/
export AZURE_API_VERSION=2023-07-01-preview
```

### 6. AWS Bedrock

```python
# AWS Bedrock configuration
model_list:
  - model_name: claude-bedrock
    litellm_params:
      model: bedrock/anthropic.claude-v2
      aws_access_key_id: your-access-key
      aws_secret_access_key: your-secret-key
      aws_region_name: us-east-1

# Environment variables
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION_NAME=us-east-1
```

## Model Configuration

### Agent-Level Configuration

```python
from jaf import Agent, ModelConfig

# Create agent with specific model configuration
agent = Agent(
    name="SpecializedAgent",
    instructions=lambda state: "You are a specialized agent.",
    tools=[],
    model_config=ModelConfig(
        name="gpt-4",              # Specific model to use
        temperature=0.7,           # Creativity/randomness (0.0-1.0)
        max_tokens=1000,          # Maximum response length
        top_p=0.9,                # Nucleus sampling
        frequency_penalty=0.0,     # Repeat token penalty
        presence_penalty=0.0       # New topic penalty
    )
)
```

### Global Configuration Override

```python
# Override model for entire conversation
config = RunConfig(
    agent_registry={"Agent": agent},
    model_provider=provider,
    model_override="claude-3-sonnet",  # Override agent's model
    max_turns=10
)
```

### Environment-Based Configuration

```python
import os

# Set default model via environment
os.environ["JAF_DEFAULT_MODEL"] = "gpt-4"
os.environ["JAF_DEFAULT_TEMPERATURE"] = "0.8"
os.environ["JAF_DEFAULT_MAX_TOKENS"] = "2000"

# Provider will use these defaults
provider = make_litellm_provider("http://localhost:4000")
```

## Advanced Features

### Tool Calling Support

JAF automatically converts your tools to the appropriate format for each model provider:

```python
from pydantic import BaseModel, Field

class CalculatorArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool:
    @property
    def schema(self):
        return type('ToolSchema', (), {
            'name': 'calculate',
            'description': 'Perform mathematical calculations',
            'parameters': CalculatorArgs
        })()
    
    async def execute(self, args: CalculatorArgs, context) -> Any:
        # Tool implementation
        pass

# JAF automatically converts this to OpenAI function format:
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"],
            "additionalProperties": false
        }
    }
}
```

### Response Format Control

```python
from jaf import Agent
from pydantic import BaseModel

class StructuredResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]

# Agent with structured output
agent = Agent(
    name="StructuredAgent",
    instructions=lambda state: "Respond with structured JSON data.",
    tools=[],
    output_codec=StructuredResponse  # Enforces JSON response format
)
```

### Streaming Support

```python
# Note: Streaming support is planned for future JAF versions
# Current implementation uses standard completion calls

class StreamingProvider:
    async def get_completion_stream(self, state, agent, config):
        """Future: Streaming completion support."""
        # Implementation for streaming responses
        pass
```

## Custom Model Providers

You can create custom model providers by implementing the `ModelProvider` protocol:

```python
from jaf.core.types import ModelProvider, RunState, Agent, RunConfig
from typing import TypeVar, Dict, Any

Ctx = TypeVar('Ctx')

class CustomModelProvider:
    """Custom model provider implementation."""
    
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    async def get_completion(
        self,
        state: RunState[Ctx],
        agent: Agent[Ctx, Any],
        config: RunConfig[Ctx]
    ) -> Dict[str, Any]:
        """Get completion from custom model service."""
        
        # Build request payload
        payload = {
            "model": agent.model_config.name if agent.model_config else "default",
            "messages": self._convert_messages(state, agent),
            "temperature": agent.model_config.temperature if agent.model_config else 0.7,
            "max_tokens": agent.model_config.max_tokens if agent.model_config else 1000
        }
        
        # Add tools if present
        if agent.tools:
            payload["tools"] = self._convert_tools(agent.tools)
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_endpoint}/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Convert response to JAF format
            return {
                'message': {
                    'content': data['choices'][0]['message']['content'],
                    'tool_calls': data['choices'][0]['message'].get('tool_calls')
                }
            }
    
    def _convert_messages(self, state: RunState[Ctx], agent: Agent[Ctx, Any]) -> List[Dict]:
        """Convert JAF messages to provider format."""
        messages = [
            {"role": "system", "content": agent.instructions(state)}
        ]
        
        for msg in state.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content,
                "tool_call_id": getattr(msg, 'tool_call_id', None)
            })
        
        return messages
    
    def _convert_tools(self, tools) -> List[Dict]:
        """Convert JAF tools to provider format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.schema.name,
                    "description": tool.schema.description,
                    "parameters": tool.schema.parameters.model_json_schema()
                }
            }
            for tool in tools
        ]

# Use custom provider
custom_provider = CustomModelProvider("https://api.custom-llm.com", "your-api-key")
```

## Performance Optimization

### Connection Pooling

```python
import httpx

class OptimizedLiteLLMProvider:
    def __init__(self, base_url: str, api_key: str):
        # Use connection pooling for better performance
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=5,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(30.0)
        )
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()
```

### Request Optimization

```python
# Optimize for specific use cases
class HighThroughputConfig:
    """Configuration optimized for high throughput."""
    temperature = 0.1        # Lower temperature for consistency
    max_tokens = 500        # Shorter responses
    top_p = 0.8            # Focus on likely tokens
    
class CreativeConfig:
    """Configuration optimized for creative tasks."""
    temperature = 0.9       # Higher temperature for creativity
    max_tokens = 2000      # Longer responses allowed
    top_p = 0.95          # More token variety
    frequency_penalty = 0.3 # Reduce repetition
```

### Caching

```python
from functools import lru_cache
import hashlib
import json

class CachedModelProvider:
    def __init__(self, base_provider):
        self.base_provider = base_provider
        self.cache = {}
    
    async def get_completion(self, state, agent, config):
        # Create cache key from request
        cache_key = self._create_cache_key(state, agent, config)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get fresh response
        response = await self.base_provider.get_completion(state, agent, config)
        
        # Cache response (be careful with memory usage)
        if len(self.cache) < 1000:  # Limit cache size
            self.cache[cache_key] = response
        
        return response
    
    def _create_cache_key(self, state, agent, config) -> str:
        """Create deterministic cache key."""
        key_data = {
            "messages": [{"role": m.role, "content": m.content} for m in state.messages],
            "agent_name": agent.name,
            "model": config.model_override or (agent.model_config.name if agent.model_config else "default"),
            "instructions": agent.instructions(state)
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
```

## Monitoring and Observability

### Request Logging

```python
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LoggingModelProvider:
    def __init__(self, base_provider):
        self.base_provider = base_provider
    
    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Log request
            logger.info(f"Model request: agent={agent.name}, messages={len(state.messages)}")
            
            response = await self.base_provider.get_completion(state, agent, config)
            
            # Log successful response
            duration = (time.time() - start_time) * 1000
            logger.info(f"Model response: duration={duration:.2f}ms, success=True")
            
            return response
            
        except Exception as e:
            # Log error
            duration = (time.time() - start_time) * 1000
            logger.error(f"Model error: duration={duration:.2f}ms, error={str(e)}")
            raise
```

### Metrics Collection

```python
from dataclasses import dataclass
from collections import defaultdict, deque
import time

@dataclass
class ModelMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    recent_durations: deque = None
    
    def __post_init__(self):
        if self.recent_durations is None:
            self.recent_durations = deque(maxlen=100)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_duration(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_duration / self.successful_requests
    
    @property
    def recent_average_duration(self) -> float:
        if not self.recent_durations:
            return 0.0
        return sum(self.recent_durations) / len(self.recent_durations)

class MetricsCollectingProvider:
    def __init__(self, base_provider):
        self.base_provider = base_provider
        self.metrics = defaultdict(ModelMetrics)
    
    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        model_name = config.model_override or (agent.model_config.name if agent.model_config else "default")
        metrics = self.metrics[model_name]
        
        start_time = time.time()
        metrics.total_requests += 1
        
        try:
            response = await self.base_provider.get_completion(state, agent, config)
            
            # Record success metrics
            duration = time.time() - start_time
            metrics.successful_requests += 1
            metrics.total_duration += duration
            metrics.recent_durations.append(duration)
            
            return response
            
        except Exception as e:
            metrics.failed_requests += 1
            raise
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all model metrics."""
        return {
            model: {
                "total_requests": metrics.total_requests,
                "success_rate": metrics.success_rate,
                "average_duration_ms": metrics.average_duration * 1000,
                "recent_average_duration_ms": metrics.recent_average_duration * 1000
            }
            for model, metrics in self.metrics.items()
        }
```

## Error Handling

### Retry Logic

```python
import asyncio
from typing import Optional

class RetryingModelProvider:
    def __init__(self, base_provider, max_retries: int = 3, base_delay: float = 1.0):
        self.base_provider = base_provider
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await self.base_provider.get_completion(state, agent, config)
                
            except Exception as e:
                last_exception = e
                
                # Don't retry on client errors (4xx)
                if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                    raise
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    logger.warning(f"Retrying model request (attempt {attempt + 1}/{self.max_retries}) after {delay}s delay")
        
        # All retries failed
        raise last_exception
```

### Fallback Providers

```python
class FallbackModelProvider:
    def __init__(self, primary_provider, fallback_provider):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
    
    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        try:
            return await self.primary_provider.get_completion(state, agent, config)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}. Falling back to secondary provider.")
            return await self.fallback_provider.get_completion(state, agent, config)

# Usage
primary = make_litellm_provider("http://localhost:4000", "primary-key")
fallback = make_litellm_provider("http://backup.company.com", "backup-key")
resilient_provider = FallbackModelProvider(primary, fallback)
```

## Best Practices

### 1. Model Selection

```python
# Choose models based on use case
MODELS = {
    "fast_chat": "gpt-3.5-turbo",        # Quick responses
    "complex_reasoning": "gpt-4",         # Complex tasks
    "code_generation": "gpt-4-turbo",     # Programming tasks
    "creative_writing": "claude-3-opus",  # Creative tasks
    "cost_optimized": "gpt-3.5-turbo",   # Budget-conscious
    "local_development": "llama2"         # Local development
}

def get_model_for_task(task_type: str) -> str:
    return MODELS.get(task_type, "gpt-3.5-turbo")
```

### 2. Configuration Management

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfiguration:
    name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    cost_per_1k_tokens: float = 0.002
    max_requests_per_minute: int = 3500
    
PREDEFINED_CONFIGS = {
    "gpt-4": ModelConfiguration("gpt-4", 0.7, 4000, 0.03, 10000),
    "gpt-3.5-turbo": ModelConfiguration("gpt-3.5-turbo", 0.7, 2000, 0.002, 3500),
    "claude-3-sonnet": ModelConfiguration("claude-3-sonnet", 0.7, 4000, 0.003, 1000)
}

def get_model_config(model_name: str) -> ModelConfiguration:
    return PREDEFINED_CONFIGS.get(model_name, ModelConfiguration(model_name))
```

### 3. Security Considerations

```python
import os
from typing import Dict

class SecureModelProvider:
    def __init__(self, provider_config: Dict[str, str]):
        # Load sensitive data from environment
        self.api_keys = {
            provider: os.getenv(f"{provider.upper()}_API_KEY")
            for provider in provider_config.keys()
        }
        
        # Validate all required keys are present
        missing_keys = [
            provider for provider, key in self.api_keys.items() 
            if key is None
        ]
        if missing_keys:
            raise ValueError(f"Missing API keys for providers: {missing_keys}")
    
    def get_provider_for_model(self, model_name: str):
        # Route to appropriate provider based on model
        if model_name.startswith("gpt"):
            return make_litellm_provider(
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=self.api_keys["openai"]
            )
        elif model_name.startswith("claude"):
            return make_litellm_provider(
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                api_key=self.api_keys["anthropic"]
            )
        # Add more providers as needed
```

## Next Steps

- Learn about [Server API](server-api.md) for HTTP endpoints
- Explore [Examples](examples.md) for real-world usage
- Check [Deployment](deployment.md) for production setup
- Review [Troubleshooting](troubleshooting.md) for common issues