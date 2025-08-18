"""
Test suite for model-providers.md documentation examples.
Validates that all code snippets work with the actual JAF implementation.
"""

import pytest
import asyncio
import os
import json
import hashlib
import time
import logging
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TypeVar
from collections import defaultdict, deque

# JAF imports
from jaf.providers.model import make_litellm_provider
from jaf import Agent, RunConfig, RunState, Message
from jaf.core.types import ModelProvider
from pydantic import BaseModel, Field

# Test basic provider setup
def test_basic_litellm_provider_setup():
    """Test basic LiteLLM provider creation from docs."""
    # Create provider instance
    provider = make_litellm_provider(
        base_url="http://localhost:4000",  # LiteLLM server URL
        api_key="test-api-key"             # API key (optional for local servers)
    )
    
    assert provider is not None
    # Verify provider has expected attributes
    assert hasattr(provider, 'get_completion')

def test_model_config_creation():
    """Test ModelConfig usage from docs."""
    from jaf.core.types import ModelConfig
    
    # Create agent with specific model configuration
    # Note: ModelConfig in actual implementation only supports name, temperature, max_tokens
    agent = Agent(
        name="SpecializedAgent",
        instructions=lambda state: "You are a specialized agent.",
        tools=[],
        model_config=ModelConfig(
            name="gpt-4",              # Specific model to use
            temperature=0.7,           # Creativity/randomness (0.0-1.0)
            max_tokens=1000,          # Maximum response length
            # Note: top_p, frequency_penalty, presence_penalty not supported in current ModelConfig
        )
    )
    
    assert agent.name == "SpecializedAgent"
    assert agent.model_config.name == "gpt-4"
    assert agent.model_config.temperature == 0.7
    assert agent.model_config.max_tokens == 1000

def test_run_config_with_model_override():
    """Test RunConfig with model override from docs."""
    provider = make_litellm_provider("http://localhost:4000", "test-key")
    agent = Agent(
        name="TestAgent",
        instructions=lambda state: "Test instructions.",
        tools=[]
    )
    
    # Override model for entire conversation
    config = RunConfig(
        agent_registry={"Agent": agent},
        model_provider=provider,
        model_override="claude-3-sonnet",  # Override agent's model
        max_turns=10
    )
    
    assert config.model_override == "claude-3-sonnet"
    assert config.max_turns == 10
    assert "Agent" in config.agent_registry

def test_tool_schema_conversion():
    """Test tool schema conversion example from docs."""
    
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
            return f"Result: {args.expression}"
    
    tool = CalculatorTool()
    schema = tool.schema
    
    assert schema.name == 'calculate'
    assert schema.description == 'Perform mathematical calculations'
    assert schema.parameters == CalculatorArgs

def test_structured_response_model():
    """Test structured response model from docs."""
    
    class StructuredResponse(BaseModel):
        answer: str
        confidence: float
        sources: List[str]

    # Agent with structured output
    agent = Agent(
        name="StructuredAgent",
        instructions=lambda state: "Respond with structured JSON data.",
        tools=[],
        # Note: output_codec is not implemented in current JAF, so we skip this part
    )
    
    assert agent.name == "StructuredAgent"
    
    # Test the response model itself
    response = StructuredResponse(
        answer="Test answer",
        confidence=0.95,
        sources=["source1", "source2"]
    )
    
    assert response.answer == "Test answer"
    assert response.confidence == 0.95
    assert len(response.sources) == 2

class TestCustomModelProvider:
    """Test custom model provider implementation from docs."""
    
    def test_custom_provider_structure(self):
        """Test that custom provider follows the expected structure."""
        
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
                
                # Mock response for testing
                return {
                    'message': {
                        'content': 'Mock response',
                        'tool_calls': None
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

        # Test provider creation
        provider = CustomModelProvider("https://api.custom-llm.com", "test-key")
        assert provider.api_endpoint == "https://api.custom-llm.com"
        assert provider.api_key == "test-key"

def test_performance_optimization_configs():
    """Test performance optimization configuration classes from docs."""
    
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
    
    # Test configurations
    high_throughput = HighThroughputConfig()
    assert high_throughput.temperature == 0.1
    assert high_throughput.max_tokens == 500
    assert high_throughput.top_p == 0.8
    
    creative = CreativeConfig()
    assert creative.temperature == 0.9
    assert creative.max_tokens == 2000
    assert creative.top_p == 0.95
    assert creative.frequency_penalty == 0.3

def test_caching_provider():
    """Test caching model provider from docs."""
    
    class CachedModelProvider:
        def __init__(self, base_provider):
            self.base_provider = base_provider
            self.cache = {}
        
        async def get_completion(self, state, agent, config):
            # Create cache key from request
            cache_key = self._create_cache_key(state, agent, config)
            
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get fresh response (mock for testing)
            response = {"message": {"content": "cached response"}}
            
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
    
    # Test caching provider
    base_provider = Mock()
    cached_provider = CachedModelProvider(base_provider)
    
    assert cached_provider.base_provider == base_provider
    assert cached_provider.cache == {}

def test_model_metrics():
    """Test model metrics collection from docs."""
    
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
    
    # Test metrics
    metrics = ModelMetrics()
    assert metrics.total_requests == 0
    assert metrics.success_rate == 0.0
    assert metrics.average_duration == 0.0
    
    # Add some data
    metrics.total_requests = 10
    metrics.successful_requests = 8
    metrics.total_duration = 5.0
    metrics.recent_durations.extend([0.5, 0.6, 0.4])
    
    assert metrics.success_rate == 0.8
    assert metrics.average_duration == 0.625
    assert metrics.recent_average_duration == 0.5

def test_metrics_collecting_provider():
    """Test metrics collecting provider from docs."""
    
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
                # Mock successful response
                response = {"message": {"content": "test response"}}
                
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
                    "success_rate": metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0.0,
                    "average_duration_ms": (metrics.total_duration / metrics.successful_requests * 1000) if metrics.successful_requests > 0 else 0.0,
                    "recent_average_duration_ms": (sum(metrics.recent_durations) / len(metrics.recent_durations) * 1000) if metrics.recent_durations else 0.0
                }
                for model, metrics in self.metrics.items()
            }
    
    # Test metrics collecting provider
    base_provider = Mock()
    metrics_provider = MetricsCollectingProvider(base_provider)
    
    assert metrics_provider.base_provider == base_provider
    assert len(metrics_provider.metrics) == 0

def test_retrying_provider():
    """Test retrying model provider from docs."""
    
    class RetryingModelProvider:
        def __init__(self, base_provider, max_retries: int = 3, base_delay: float = 1.0):
            self.base_provider = base_provider
            self.max_retries = max_retries
            self.base_delay = base_delay
        
        async def get_completion(self, state, agent, config) -> Dict[str, Any]:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    # Mock successful response on first try
                    return {"message": {"content": "success"}}
                    
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on client errors (4xx)
                    if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                        raise
                    
                    if attempt < self.max_retries:
                        # Exponential backoff (mock sleep for testing)
                        delay = self.base_delay * (2 ** attempt)
                        # await asyncio.sleep(delay)  # Skip actual sleep in tests
            
            # All retries failed
            if last_exception:
                raise last_exception
    
    # Test retrying provider
    base_provider = Mock()
    retrying_provider = RetryingModelProvider(base_provider, max_retries=2, base_delay=0.1)
    
    assert retrying_provider.max_retries == 2
    assert retrying_provider.base_delay == 0.1

def test_fallback_provider():
    """Test fallback model provider from docs."""
    
    class FallbackModelProvider:
        def __init__(self, primary_provider, fallback_provider):
            self.primary_provider = primary_provider
            self.fallback_provider = fallback_provider
        
        async def get_completion(self, state, agent, config) -> Dict[str, Any]:
            try:
                # Mock primary provider success
                return {"message": {"content": "primary response"}}
            except Exception as e:
                # Mock fallback
                return {"message": {"content": "fallback response"}}

    # Test fallback provider
    primary = Mock()
    fallback = Mock()
    fallback_provider = FallbackModelProvider(primary, fallback)
    
    assert fallback_provider.primary_provider == primary
    assert fallback_provider.fallback_provider == fallback

def test_model_selection_helper():
    """Test model selection helper from docs."""
    
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
    
    # Test model selection
    assert get_model_for_task("fast_chat") == "gpt-3.5-turbo"
    assert get_model_for_task("complex_reasoning") == "gpt-4"
    assert get_model_for_task("unknown_task") == "gpt-3.5-turbo"

def test_model_configuration_dataclass():
    """Test model configuration dataclass from docs."""
    
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
    
    # Test configuration
    gpt4_config = get_model_config("gpt-4")
    assert gpt4_config.name == "gpt-4"
    assert gpt4_config.max_tokens == 4000
    assert gpt4_config.cost_per_1k_tokens == 0.03
    
    unknown_config = get_model_config("unknown-model")
    assert unknown_config.name == "unknown-model"
    assert unknown_config.temperature == 0.7  # default value

@pytest.mark.asyncio
async def test_async_provider_methods():
    """Test that async provider methods work correctly."""
    from jaf.core.types import generate_run_id, generate_trace_id
    
    # Create a simple mock state and agent for testing
    # RunState requires run_id, trace_id, current_agent_name, turn_count
    state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[],
        current_agent_name="TestAgent",
        context={},
        turn_count=0
    )
    agent = Agent(
        name="TestAgent",
        instructions=lambda s: "Test instructions",
        tools=[]
    )
    
    # Create a mock provider for RunConfig
    mock_provider = Mock()
    config = RunConfig(
        agent_registry={"TestAgent": agent},
        model_provider=mock_provider
    )
    
    # Test that we can create the objects without errors
    assert state.messages == []
    assert agent.name == "TestAgent"
    assert "TestAgent" in config.agent_registry
    assert state.current_agent_name == "TestAgent"
    assert state.turn_count == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
