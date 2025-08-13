# Callback System - Advanced Agent Instrumentation

!!! info "Revolutionary Agent Control"
    The ADK Callback System transforms JAF from a simple agent executor into a sophisticated, observable state machine with complete control over every aspect of agent execution.

## 🎯 Overview

The Callback System enables advanced agent patterns by providing **14+ hooks** that instrument every critical stage of agent execution. This allows developers to implement sophisticated behaviors like:

- **ReAct Patterns** - Iterative reasoning with synthesis checking
- **Dynamic Query Refinement** - Based on accumulated context
- **Loop Detection** - Preventing repetitive behaviors
- **Custom LLM Strategies** - Message modification and call skipping
- **Context Accumulation** - Intelligent information gathering

## 🔧 Core Concepts

### RunnerCallbacks Protocol

The `RunnerCallbacks` protocol defines hooks for instrumenting agent execution:

```python
from adk.runners import RunnerCallbacks, RunnerConfig, execute_agent
from typing import Optional, List, Dict, Any

class MyCallbacks:
    """Custom callback implementation."""
    
    # Lifecycle hooks
    async def on_start(self, context, message, session_state):
        """Called at agent execution start."""
        print(f"🚀 Processing: {message.content}")
    
    async def on_complete(self, response):
        """Called when execution completes successfully."""
        print(f"✅ Completed in {response.execution_time_ms}ms")
    
    async def on_error(self, error, context):
        """Called when execution encounters an error."""
        print(f"❌ Error: {error}")
    
    # LLM interaction hooks
    async def on_before_llm_call(self, agent, message, session_state):
        """Modify or skip LLM calls."""
        # Example: Add context to message
        enriched_content = f"Context: {session_state.get('context', '')}\n{message.content}"
        return {'message': Message(role='user', content=enriched_content)}
    
    async def on_after_llm_call(self, response, session_state):
        """Modify LLM responses."""
        # Example: Post-process response
        if len(response.content) < 50:
            enhanced = f"{response.content}\n\n[Response enhanced for completeness]"
            return Message(role='assistant', content=enhanced)
        return None
```

### RunnerConfig Enhancement

Configure agents with callback support:

```python
from adk.runners import RunnerConfig
from jaf.core.types import Agent

# Create agent with callback-enabled runner
config = RunnerConfig(
    agent=my_agent,
    session_provider=session_provider,
    callbacks=MyCallbacks(),
    
    # Advanced settings
    max_llm_calls=10,
    enable_context_accumulation=True,
    enable_loop_detection=True,
    max_context_items=100,
    max_repeated_tools=3
)

# Execute with full instrumentation
result = await execute_agent(config, session_state, message, context, model_provider)
```

## 🎣 Available Hooks

### 1. Lifecycle Hooks

Control the overall execution lifecycle:

```python
class LifecycleCallbacks:
    async def on_start(self, context, message, session_state):
        """Execution started - initialize tracking."""
        self.start_time = time.time()
        self.query_id = generate_id()
        
    async def on_complete(self, response):
        """Execution completed - log metrics."""
        duration = time.time() - self.start_time
        self.log_metrics(self.query_id, duration, response)
        
    async def on_error(self, error, context):
        """Handle execution errors gracefully."""
        self.log_error(self.query_id, error, context)
```

### 2. LLM Interaction Hooks

Complete control over LLM interactions:

```python
class LLMControlCallbacks:
    async def on_before_llm_call(self, agent, message, session_state):
        """Modify messages before LLM call."""
        # Skip LLM for cached responses
        cached_response = self.check_cache(message.content)
        if cached_response:
            return {'skip': True, 'response': cached_response}
        
        # Enrich message with context
        context_summary = self.get_context_summary(session_state)
        enriched_message = self.add_context(message, context_summary)
        return {'message': enriched_message}
    
    async def on_after_llm_call(self, response, session_state):
        """Post-process LLM responses."""
        # Cache response for future use
        self.cache_response(response)
        
        # Apply post-processing rules
        return self.apply_formatting_rules(response)
```

### 3. Iteration Control Hooks

Implement sophisticated reasoning loops:

```python
class IterativeReasoningCallbacks:
    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        
    async def on_iteration_start(self, iteration):
        """Control iteration flow."""
        self.iteration_count = iteration
        print(f"🔄 Iteration {iteration}/{self.max_iterations}")
        
        if iteration > self.max_iterations:
            return {'continue_iteration': False}
        return None
    
    async def on_iteration_complete(self, iteration, has_tool_calls):
        """Decide whether to continue iterating."""
        if not has_tool_calls:
            # No tools called, likely finished
            return {'should_stop': True}
        
        if self.sufficient_information_gathered():
            return {'should_stop': True}
        
        return {'should_continue': True}
```

### 4. Tool Execution Hooks

Fine-grained tool control:

```python
class ToolControlCallbacks:
    async def on_before_tool_selection(self, tools, context_data):
        """Filter or modify available tools."""
        # Limit tools based on context
        if len(context_data) > 10:
            # Only allow synthesis tools when we have enough data
            synthesis_tools = [t for t in tools if 'synthesis' in t.schema.name]
            return {'tools': synthesis_tools}
        return None
    
    async def on_tool_selected(self, tool_name, params):
        """Track tool usage."""
        self.log_tool_selection(tool_name, params)
    
    async def on_before_tool_execution(self, tool, params):
        """Modify parameters or skip execution."""
        # Add authentication to API calls
        if tool.schema.name == 'api_call':
            enhanced_params = {**params, 'auth_token': self.get_auth_token()}
            return {'params': enhanced_params}
        return None
    
    async def on_after_tool_execution(self, tool, result, error=None):
        """Process tool results."""
        if error:
            self.handle_tool_error(tool, error)
            return None
        
        # Transform result format
        return self.standardize_result_format(result)
```

### 5. Synthesis and Context Hooks

Enable ReAct-style patterns:

```python
class SynthesisCallbacks:
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        self.context_accumulator = []
    
    async def on_check_synthesis(self, session_state, context_data):
        """Determine if synthesis is complete."""
        if len(context_data) < 3:
            return None  # Need more information
        
        # Analyze information completeness
        coverage_score = self.analyze_coverage(context_data)
        quality_score = self.analyze_quality(context_data)
        confidence = (coverage_score + quality_score) / 2
        
        if confidence >= self.confidence_threshold:
            synthesis_prompt = self.create_synthesis_prompt(context_data)
            return {
                'complete': True,
                'answer': synthesis_prompt,
                'confidence': confidence
            }
        return None
    
    async def on_query_rewrite(self, original_query, context_data):
        """Refine queries based on accumulated context."""
        gaps = self.identify_knowledge_gaps(context_data)
        if gaps:
            return f"{original_query} focusing on {', '.join(gaps)}"
        return None
    
    async def on_context_update(self, current_context, new_items):
        """Manage context accumulation."""
        # Deduplicate and filter
        filtered_items = self.deduplicate_and_filter(new_items)
        
        # Merge with existing context
        merged_context = current_context + filtered_items
        
        # Sort by relevance and limit size
        sorted_context = sorted(merged_context, key=lambda x: x.get('relevance', 0), reverse=True)
        return sorted_context[:50]  # Keep top 50 items
```

### 6. Loop Detection and Prevention

Prevent repetitive behaviors:

```python
class LoopDetectionCallbacks:
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.tool_history = []
    
    async def on_loop_detection(self, tool_history, current_tool):
        """Detect and prevent loops."""
        if len(tool_history) < 3:
            return False
        
        # Check for repetitive tool calls
        recent_tools = [item['tool'] for item in tool_history[-3:]]
        if recent_tools.count(current_tool) > 2:
            print(f"🚫 Loop detected: {current_tool} called repeatedly")
            return True
        
        # Check for parameter similarity
        recent_params = [item.get('params', {}) for item in tool_history[-3:]]
        for params in recent_params:
            if self.calculate_similarity(params, current_tool) > self.similarity_threshold:
                print(f"🚫 Similar parameters detected for {current_tool}")
                return True
        
        return False
```

## 🚀 Advanced Patterns

### ReAct (Reasoning + Acting) Pattern

Implement sophisticated reasoning loops:

```python
class ReActAgent:
    """ReAct pattern implementation using callbacks."""
    
    def __init__(self):
        self.observations = []
        self.thoughts = []
        self.actions = []
    
    async def on_iteration_start(self, iteration):
        """Think about what to do next."""
        if iteration == 1:
            thought = f"I need to gather information about the user's query."
        else:
            thought = f"Based on {len(self.observations)} observations, I should..."
        
        self.thoughts.append(thought)
        print(f"🤔 Thought: {thought}")
        return None
    
    async def on_before_tool_execution(self, tool, params):
        """Record planned action."""
        action = f"Using {tool.schema.name} with {params}"
        self.actions.append(action)
        print(f"🎯 Action: {action}")
        return None
    
    async def on_after_tool_execution(self, tool, result, error=None):
        """Record observation."""
        if error:
            observation = f"Action failed: {error}"
        else:
            observation = f"Observed: {result.get('summary', str(result)[:100])}"
        
        self.observations.append(observation)
        print(f"👁️ Observation: {observation}")
        return None
    
    async def on_check_synthesis(self, session_state, context_data):
        """Decide if we have enough information."""
        if len(self.observations) >= 3:
            final_thought = "I have sufficient information to provide a comprehensive answer."
            synthesis = self.synthesize_observations()
            
            return {
                'complete': True,
                'answer': f"Final thought: {final_thought}\n\nAnswer: {synthesis}",
                'confidence': 0.9
            }
        return None
```

### Intelligent Caching Pattern

Implement smart caching with callbacks:

```python
class CachingCallbacks:
    def __init__(self):
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def on_before_llm_call(self, agent, message, session_state):
        """Check cache before LLM call."""
        cache_key = self.generate_cache_key(message, session_state)
        
        if cache_key in self.cache:
            self.cache_hits += 1
            cached_response = self.cache[cache_key]
            print(f"💾 Cache hit! Skipping LLM call")
            return {'skip': True, 'response': cached_response}
        
        self.cache_misses += 1
        return None
    
    async def on_after_llm_call(self, response, session_state):
        """Cache LLM response."""
        cache_key = self.generate_cache_key(response, session_state)
        self.cache[cache_key] = response
        
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
        print(f"📊 Cache hit rate: {hit_rate:.1f}%")
        return None
```

### Multi-Agent Coordination

Coordinate multiple agents with callbacks:

```python
class CoordinationCallbacks:
    def __init__(self, agent_registry):
        self.agent_registry = agent_registry
        self.delegation_history = []
    
    async def on_before_tool_selection(self, tools, context_data):
        """Delegate to specialized agents."""
        query_type = self.classify_query(context_data)
        
        if query_type == 'technical':
            specialist_agent = self.agent_registry['TechnicalExpert']
            return {'custom_selection': {
                'tool': 'delegate_to_agent',
                'params': {'agent': specialist_agent, 'context': context_data}
            }}
        
        return None
    
    async def on_tool_selected(self, tool_name, params):
        """Track delegation decisions."""
        if tool_name == 'delegate_to_agent':
            self.delegation_history.append({
                'agent': params['agent'],
                'reason': 'Specialized expertise required',
                'timestamp': time.time()
            })
```

## 📊 Performance and Debugging

### Performance Monitoring

Track execution metrics with callbacks:

```python
class PerformanceCallbacks:
    def __init__(self):
        self.metrics = {
            'llm_calls': 0,
            'tool_calls': 0,
            'total_tokens': 0,
            'cache_hits': 0
        }
    
    async def on_before_llm_call(self, agent, message, session_state):
        self.metrics['llm_calls'] += 1
        return None
    
    async def on_tool_selected(self, tool_name, params):
        self.metrics['tool_calls'] += 1
        return None
    
    async def on_complete(self, response):
        print(f"📊 Performance Metrics:")
        print(f"   LLM Calls: {self.metrics['llm_calls']}")
        print(f"   Tool Calls: {self.metrics['tool_calls']}")
        print(f"   Execution Time: {response.execution_time_ms}ms")
```

### Debug Logging

Comprehensive debug logging:

```python
class DebugCallbacks:
    def __init__(self, log_level='INFO'):
        self.log_level = log_level
        self.debug_info = []
    
    async def on_iteration_start(self, iteration):
        self.log(f"🔄 Starting iteration {iteration}")
        return None
    
    async def on_before_llm_call(self, agent, message, session_state):
        self.log(f"🤖 LLM Call: {message.content[:100]}...")
        return None
    
    async def on_after_tool_execution(self, tool, result, error=None):
        if error:
            self.log(f"❌ Tool {tool.schema.name} failed: {error}")
        else:
            self.log(f"✅ Tool {tool.schema.name} succeeded")
        return None
    
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.debug_info.append(log_entry)
        if self.log_level == 'DEBUG':
            print(log_entry)
```

## 🧪 Testing Callbacks

### Unit Testing

Test individual callbacks:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_synthesis_callback():
    """Test synthesis checking logic."""
    callbacks = SynthesisCallbacks(confidence_threshold=0.8)
    
    # Test with insufficient data
    result = await callbacks.on_check_synthesis({}, [])
    assert result is None
    
    # Test with sufficient high-quality data
    context_data = [
        {'content': 'High quality content 1', 'relevance': 0.9},
        {'content': 'High quality content 2', 'relevance': 0.85},
        {'content': 'High quality content 3', 'relevance': 0.8}
    ]
    result = await callbacks.on_check_synthesis({}, context_data)
    assert result['complete'] is True
    assert result['confidence'] >= 0.8

@pytest.mark.asyncio
async def test_loop_detection():
    """Test loop detection logic."""
    callbacks = LoopDetectionCallbacks()
    
    # No loop for different tools
    tool_history = [
        {'tool': 'search', 'params': {'q': 'query1'}},
        {'tool': 'analyze', 'params': {'data': 'data1'}}
    ]
    result = await callbacks.on_loop_detection(tool_history, 'summarize')
    assert result is False
    
    # Loop detected for repeated tools
    tool_history = [
        {'tool': 'search', 'params': {'q': 'query1'}},
        {'tool': 'search', 'params': {'q': 'query2'}},
        {'tool': 'search', 'params': {'q': 'query3'}}
    ]
    result = await callbacks.on_loop_detection(tool_history, 'search')
    assert result is True
```

### Integration Testing

Test complete callback workflows:

```python
@pytest.mark.asyncio
async def test_iterative_workflow():
    """Test complete iterative agent workflow."""
    
    class TestCallbacks:
        def __init__(self):
            self.iterations = 0
            self.context_items = []
        
        async def on_iteration_start(self, iteration):
            self.iterations = iteration
            return None
        
        async def on_context_update(self, current_context, new_items):
            self.context_items.extend(new_items)
            return self.context_items
        
        async def on_check_synthesis(self, session_state, context_data):
            if len(context_data) >= 3:
                return {'complete': True, 'answer': 'Test synthesis'}
            return None
    
    callbacks = TestCallbacks()
    config = RunnerConfig(
        agent=test_agent,
        callbacks=callbacks,
        enable_context_accumulation=True
    )
    
    # Mock context data accumulation
    result = await execute_agent(config, {}, test_message, {}, mock_provider)
    
    assert callbacks.iterations > 0
    assert len(callbacks.context_items) >= 3
    assert 'Test synthesis' in result.content.content
```

## 🔗 Integration Examples

### With JAF Core

```python
from jaf.core.types import Agent, Message
from adk.runners import RunnerConfig, execute_agent

# Create JAF agent
def agent_instructions(state):
    return "You are a research assistant with iterative capabilities."

agent = Agent(
    name="ResearchAgent",
    instructions=agent_instructions,
    tools=[search_tool, analyze_tool]
)

# Add callback-based behavior
class ResearchCallbacks:
    async def on_query_rewrite(self, original_query, context_data):
        return self.refine_research_query(original_query, context_data)

# Configure and execute
config = RunnerConfig(agent=agent, callbacks=ResearchCallbacks())
result = await execute_agent(config, session_state, message, context, provider)
```

### With Memory System

```python
from jaf.memory import create_in_memory_provider, MemoryConfig

# Integrate callbacks with memory
class MemoryAwareCallbacks:
    async def on_start(self, context, message, session_state):
        # Load relevant memories
        memories = await self.memory_provider.search_memories(message.content)
        session_state['relevant_memories'] = memories
    
    async def on_complete(self, response):
        # Store successful interactions
        await self.memory_provider.store_interaction(response)

memory_provider = create_in_memory_provider()
callbacks = MemoryAwareCallbacks()
callbacks.memory_provider = memory_provider

config = RunnerConfig(agent=agent, callbacks=callbacks)
```

## 🎯 Best Practices

### 1. Callback Design Principles

- **Single Responsibility**: Each callback should have one clear purpose
- **Error Resilience**: Handle exceptions gracefully to avoid breaking execution
- **Performance Awareness**: Keep callbacks lightweight for production use
- **State Management**: Use instance variables to maintain state across callbacks

### 2. Common Patterns

```python
# ✅ Good: Clear, focused callback
async def on_start(self, context, message, session_state):
    """Initialize tracking for this execution."""
    self.start_time = time.time()
    self.execution_id = generate_unique_id()

# ❌ Avoid: Callback doing too much
async def on_start(self, context, message, session_state):
    """DON'T: Multiple responsibilities in one callback."""
    self.start_time = time.time()
    self.validate_input(message)  # Should be separate
    self.load_user_preferences(context)  # Should be separate
    self.initialize_caching()  # Should be separate
```

### 3. Error Handling

```python
class RobustCallbacks:
    async def on_before_llm_call(self, agent, message, session_state):
        try:
            return self.enhance_message(message, session_state)
        except Exception as e:
            # Log error but don't break execution
            self.log_error(f"Message enhancement failed: {e}")
            return None  # Let execution continue normally
```

### 4. Testing Strategy

- **Unit Test**: Individual callback methods
- **Integration Test**: Complete callback workflows
- **Performance Test**: Ensure minimal overhead
- **Error Test**: Verify graceful failure handling

## 🔮 Advanced Use Cases

### Real-time Monitoring

```python
class MonitoringCallbacks:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
    
    async def on_start(self, context, message, session_state):
        self.metrics.increment('agent.executions.started')
    
    async def on_complete(self, response):
        self.metrics.increment('agent.executions.completed')
        self.metrics.histogram('agent.execution.duration', response.execution_time_ms)
    
    async def on_error(self, error, context):
        self.metrics.increment('agent.executions.failed')
        self.metrics.increment(f'agent.errors.{type(error).__name__}')
```

### A/B Testing

```python
class ABTestingCallbacks:
    def __init__(self, experiment_config):
        self.experiment = experiment_config
    
    async def on_before_llm_call(self, agent, message, session_state):
        if self.experiment.should_test(session_state.get('user_id')):
            # Use experimental prompt template
            enhanced_message = self.experiment.apply_variant(message)
            return {'message': enhanced_message}
        return None
```

### Content Filtering

```python
class ContentFilterCallbacks:
    def __init__(self, filter_rules):
        self.filter_rules = filter_rules
    
    async def on_after_llm_call(self, response, session_state):
        if not self.filter_rules.is_safe(response.content):
            safe_response = self.filter_rules.sanitize(response.content)
            return Message(role='assistant', content=safe_response)
        return None
```

---

!!! tip "Getting Started"
    Start with simple lifecycle callbacks (`on_start`, `on_complete`) and gradually add more sophisticated hooks as you need advanced behaviors. The callback system is designed to be incrementally adoptable.

!!! warning "Performance Considerations"
    While callbacks add minimal overhead, avoid heavy computation in frequently called hooks like `on_before_llm_call`. Consider using async operations and caching for expensive operations.

!!! example "Complete Example"
    See the [Iterative Search Agent Example](https://github.com/your-repo/examples/iterative_search_agent.py) for a comprehensive demonstration of advanced callback patterns in action.