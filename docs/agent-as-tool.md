# Agent-as-Tool: Hierarchical Agent Orchestration

JAF's agent-as-tool functionality enables sophisticated hierarchical agent architectures where specialized agents can be used as tools by other agents. This powerful pattern allows for modular, reusable, and scalable multi-agent systems.

## Overview

The agent-as-tool pattern transforms any JAF agent into a tool that can be used by other agents, creating hierarchical orchestration patterns. This enables:

- **Specialized Expertise**: Delegate specific tasks to expert agents
- **Modular Architecture**: Build complex systems from composable components
- **Conditional Execution**: Enable/disable agent tools based on context
- **Session Management**: Control memory and state sharing between agents
- **Hierarchical Reasoning**: Create supervisor-worker agent patterns

### Key Concepts

- **Parent Agent**: The orchestrating agent that uses other agents as tools
- **Child Agent**: The specialized agent that executes as a tool
- **Context Inheritance**: How context and configuration flow between agents
- **Session Preservation**: Whether child agents share parent's memory/session
- **Conditional Enabling**: Dynamic tool availability based on context

## Quick Start

### Basic Agent-as-Tool Example

```python
import asyncio
from dataclasses import dataclass
from jaf import Agent, ModelConfig, RunConfig, RunState, Message, run
from jaf.core.types import ContentRole, generate_run_id, generate_trace_id
from jaf.providers.model import make_litellm_provider

@dataclass(frozen=True)
class TranslationContext:
    user_id: str
    target_languages: list[str]

# Create specialized translation agents
spanish_agent = Agent(
    name="spanish_translator",
    instructions=lambda state: "Translate the user's message to Spanish. Reply only with the Spanish translation.",
    model_config=ModelConfig(name="gpt-4", temperature=0.3)
)

french_agent = Agent(
    name="french_translator", 
    instructions=lambda state: "Translate the user's message to French. Reply only with the French translation.",
    model_config=ModelConfig(name="gpt-4", temperature=0.3)
)

# Convert agents to tools
spanish_tool = spanish_agent.as_tool(
    tool_name="translate_to_spanish",
    tool_description="Translate text to Spanish",
    max_turns=3
)

french_tool = french_agent.as_tool(
    tool_name="translate_to_french", 
    tool_description="Translate text to French",
    max_turns=3
)

# Create orchestrator agent
orchestrator = Agent(
    name="translation_orchestrator",
    instructions=lambda state: (
        "You are a translation coordinator. Use your translation tools to provide "
        "translations in the requested languages. Always use the appropriate tools."
    ),
    tools=[spanish_tool, french_tool],
    model_config=ModelConfig(name="gpt-4", temperature=0.1)
)

async def main():
    config = RunConfig(
        agent_registry={"translation_orchestrator": orchestrator},
        model_provider=make_litellm_provider(
            base_url="http://localhost:4000",
            api_key="your-api-key"
        ),
        max_turns=10
    )
    
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content="Translate 'Hello, how are you?' to Spanish and French")],
        current_agent_name="translation_orchestrator",
        context=TranslationContext(
            user_id="user123",
            target_languages=["spanish", "french"]
        ),
        turn_count=0
    )
    
    result = await run(initial_state, config)
    print(f"Result: {result.outcome.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Creating Agent Tools

### The `as_tool()` Method

Every JAF agent has an `as_tool()` method that converts it into a tool:

```python
agent_tool = agent.as_tool(
    tool_name="custom_tool_name",           # Optional: custom tool name
    tool_description="Tool description",    # Optional: custom description  
    max_turns=5,                           # Optional: limit agent turns
    custom_output_extractor=None,          # Optional: custom output processing
    is_enabled=True,                       # Optional: conditional enabling
    metadata={"category": "translation"},  # Optional: tool metadata
    timeout=30.0,                         # Optional: execution timeout
    preserve_session=False                 # Optional: session inheritance
)
```

### Tool Parameters

#### `tool_name` and `tool_description`
Customize how the tool appears to the parent agent:

```python
# Default naming (based on agent name)
default_tool = spanish_agent.as_tool()
# Tool name: "run_spanish_translator"

# Custom naming
custom_tool = spanish_agent.as_tool(
    tool_name="translate_spanish",
    tool_description="Translate any text to Spanish with high accuracy"
)
```

#### `max_turns`
Limit the number of turns the child agent can take:

```python
# Quick translation (limit turns for efficiency)
quick_tool = translator_agent.as_tool(max_turns=2)

# Complex reasoning (allow more turns)
research_tool = researcher_agent.as_tool(max_turns=20)
```

#### `custom_output_extractor`
Process the agent's output before returning to parent:

```python
from jaf.core.agent_tool import create_json_output_extractor, create_default_output_extractor

# Extract JSON from agent output
json_tool = data_agent.as_tool(
    custom_output_extractor=create_json_output_extractor()
)

# Custom extraction logic
def extract_summary(run_result):
    """Extract just the summary from agent output."""
    if run_result.outcome.status == 'completed':
        output = run_result.outcome.output
        # Extract summary section
        if "Summary:" in output:
            return output.split("Summary:", 1)[1].strip()
        return output
    return "Agent execution failed"

summary_tool = analysis_agent.as_tool(
    custom_output_extractor=extract_summary
)

# Async output extractor
async def async_extractor(run_result):
    """Async output processing."""
    output = run_result.outcome.output
    # Perform async processing
    processed = await some_async_function(output)
    return processed

async_tool = agent.as_tool(
    custom_output_extractor=async_extractor
)
```

#### `timeout`
Set execution timeout for agent tools:

```python
# Fast operations
quick_tool = search_agent.as_tool(timeout=10.0)

# Long-running operations  
analysis_tool = deep_analysis_agent.as_tool(timeout=120.0)
```

## Conditional Tool Enabling

### Static Enabling
Simple boolean control:

```python
# Always enabled
always_tool = agent.as_tool(is_enabled=True)

# Always disabled  
disabled_tool = agent.as_tool(is_enabled=False)
```

### Dynamic Enabling with Functions
Enable tools based on context:

```python
def premium_user_only(context, agent):
    """Enable tool only for premium users."""
    return context.user_type == "premium"

def business_hours_only(context, agent):
    """Enable tool only during business hours."""
    from datetime import datetime
    current_hour = datetime.now().hour
    return 9 <= current_hour <= 17

def language_specific(target_language):
    """Enable tool only for specific language."""
    def enabler(context, agent):
        return target_language in context.target_languages
    return enabler

# Usage
premium_tool = expensive_agent.as_tool(
    is_enabled=premium_user_only
)

support_tool = human_support_agent.as_tool(
    is_enabled=business_hours_only
)

spanish_tool = spanish_agent.as_tool(
    is_enabled=language_specific("spanish")
)
```

### Async Enabling Functions
For complex async validation:

```python
async def check_api_quota(context, agent):
    """Check if user has API quota remaining."""
    quota_service = get_quota_service()
    remaining = await quota_service.get_remaining_quota(context.user_id)
    return remaining > 0

async def validate_permissions(context, agent):
    """Validate user permissions asynchronously."""
    auth_service = get_auth_service()
    permissions = await auth_service.get_user_permissions(context.user_id)
    return "advanced_tools" in permissions

# Usage
quota_tool = api_agent.as_tool(
    is_enabled=check_api_quota
)

admin_tool = admin_agent.as_tool(
    is_enabled=validate_permissions
)
```

### Convenience Functions
JAF provides helper functions for common patterns:

```python
from jaf.core.agent_tool import create_conditional_enabler

# Context attribute checking
permission_enabler = create_conditional_enabler("has_permission", True)
language_enabler = create_conditional_enabler("preferred_language", "spanish")

permission_tool = agent.as_tool(is_enabled=permission_enabler)
spanish_tool = agent.as_tool(is_enabled=language_enabler)
```

## Session Management

### Session Preservation Options

Control how child agents inherit parent session state:

```python
# Ephemeral execution (default: preserve_session=False)
# Child agent gets fresh session, no shared memory
ephemeral_tool = agent.as_tool(preserve_session=False)

# Shared session (preserve_session=True)  
# Child agent shares parent's conversation_id and memory
shared_tool = agent.as_tool(preserve_session=True)
```

### Use Cases for Session Preservation

#### Ephemeral Sessions (Default)
Best for independent, stateless operations:

```python
# Translation doesn't need conversation history
translator_tool = translator_agent.as_tool(preserve_session=False)

# Data analysis on isolated inputs
analyzer_tool = data_agent.as_tool(preserve_session=False)

# One-off calculations
calculator_tool = calc_agent.as_tool(preserve_session=False)
```

#### Shared Sessions
Best for context-aware operations:

```python
# Customer service agent that needs conversation history
support_tool = support_agent.as_tool(preserve_session=True)

# Personal assistant that builds on previous interactions
assistant_tool = personal_agent.as_tool(preserve_session=True)

# Research agent that accumulates knowledge
research_tool = research_agent.as_tool(preserve_session=True)
```

### Memory Provider Integration

Session preservation works with all memory providers:

```python
from jaf.providers.memory import RedisMemoryProvider

# Configure memory provider
memory_provider = RedisMemoryProvider(
    host="localhost",
    port=6379,
    db=0
)

config = RunConfig(
    agent_registry=agents,
    model_provider=model_provider,
    memory=memory_provider,
    conversation_id="user_123_session"
)

# Shared session tools will use the same Redis storage
shared_tool = agent.as_tool(preserve_session=True)
```

## Advanced Patterns

### Multi-Level Hierarchies

Create deep agent hierarchies:

```python
# Level 3: Specialized processors
tokenizer_agent = Agent(name="tokenizer", instructions=tokenizer_instructions)
parser_agent = Agent(name="parser", instructions=parser_instructions)
validator_agent = Agent(name="validator", instructions=validator_instructions)

# Level 2: Processing coordinator  
processor_agent = Agent(
    name="processor",
    instructions=processor_instructions,
    tools=[
        tokenizer_agent.as_tool(),
        parser_agent.as_tool(), 
        validator_agent.as_tool()
    ]
)

# Level 1: Main orchestrator
main_agent = Agent(
    name="orchestrator",
    instructions=orchestrator_instructions,
    tools=[processor_agent.as_tool()]
)
```

### Conditional Tool Chains

Enable tool chains based on context:

```python
@dataclass(frozen=True)
class ProcessingContext:
    user_id: str
    processing_level: str  # "basic", "advanced", "expert"
    available_credits: int

def basic_enabled(context, agent):
    return context.processing_level in ["basic", "advanced", "expert"]

def advanced_enabled(context, agent):
    return context.processing_level in ["advanced", "expert"]

def expert_enabled(context, agent):
    return context.processing_level == "expert" and context.available_credits > 100

orchestrator = Agent(
    name="smart_processor",
    instructions=lambda state: "Use appropriate processing tools based on user level",
    tools=[
        basic_processor.as_tool(is_enabled=basic_enabled),
        advanced_processor.as_tool(is_enabled=advanced_enabled),
        expert_processor.as_tool(is_enabled=expert_enabled)
    ]
)
```

### Error Handling and Fallbacks

Handle agent tool failures gracefully:

```python
def create_fallback_chain(primary_agent, fallback_agent):
    """Create a tool that tries primary first, then fallback."""
    
    def smart_enabler(context, agent):
        # Primary is always enabled, fallback only if primary unavailable
        if agent.name == primary_agent.name:
            return True
        # Enable fallback only in certain conditions
        return context.use_fallback or not context.primary_available
    
    return [
        primary_agent.as_tool(
            tool_name="primary_processor",
            is_enabled=smart_enabler
        ),
        fallback_agent.as_tool(
            tool_name="fallback_processor", 
            is_enabled=smart_enabler
        )
    ]

# Usage
tools = create_fallback_chain(gpt4_agent, gpt3_agent)
orchestrator = Agent(name="robust_processor", tools=tools)
```

### Agent Tool Composition

Combine multiple agent tools for complex workflows:

```python
class WorkflowContext:
    def __init__(self, user_id: str, workflow_type: str):
        self.user_id = user_id
        self.workflow_type = workflow_type
        self.steps_completed = []

def workflow_step_enabler(step_name):
    """Enable tool only if previous steps completed."""
    def enabler(context: WorkflowContext, agent):
        required_steps = {
            "analyze": [],
            "process": ["analyze"], 
            "validate": ["analyze", "process"],
            "finalize": ["analyze", "process", "validate"]
        }
        
        required = required_steps.get(step_name, [])
        return all(step in context.steps_completed for step in required)
    
    return enabler

workflow_agent = Agent(
    name="workflow_orchestrator",
    instructions=lambda state: "Execute workflow steps in correct order",
    tools=[
        analyzer_agent.as_tool(
            tool_name="analyze_data",
            is_enabled=workflow_step_enabler("analyze")
        ),
        processor_agent.as_tool(
            tool_name="process_data", 
            is_enabled=workflow_step_enabler("process")
        ),
        validator_agent.as_tool(
            tool_name="validate_results",
            is_enabled=workflow_step_enabler("validate")
        ),
        finalizer_agent.as_tool(
            tool_name="finalize_output",
            is_enabled=workflow_step_enabler("finalize") 
        )
    ]
)
```

## Production Patterns

### Agent Registry Management

Organize agents and tools for large systems:

```python
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AgentToolRegistry:
    """Centralized registry for agent tools."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tool_configs: Dict[str, dict] = {}
    
    def register_agent(self, agent: Agent, tool_config: dict = None):
        """Register an agent with optional tool configuration."""
        self.agents[agent.name] = agent
        if tool_config:
            self.tool_configs[agent.name] = tool_config
    
    def create_tool(self, agent_name: str, **overrides):
        """Create tool from registered agent with overrides."""
        agent = self.agents[agent_name]
        config = self.tool_configs.get(agent_name, {})
        config.update(overrides)
        return agent.as_tool(**config)
    
    def create_orchestrator(self, name: str, instructions, enabled_tools: List[str]):
        """Create orchestrator with selected tools."""
        tools = [self.create_tool(tool_name) for tool_name in enabled_tools]
        return Agent(name=name, instructions=instructions, tools=tools)

# Usage
registry = AgentToolRegistry()

# Register specialized agents
registry.register_agent(
    spanish_translator,
    {"tool_name": "translate_spanish", "max_turns": 3}
)

registry.register_agent(
    french_translator,
    {"tool_name": "translate_french", "max_turns": 3}
)

registry.register_agent(
    data_analyzer,
    {"tool_name": "analyze_data", "timeout": 60.0}
)

# Create orchestrators dynamically
translation_agent = registry.create_orchestrator(
    "translator",
    translation_instructions,
    ["spanish_translator", "french_translator"]
)

analysis_agent = registry.create_orchestrator(
    "analyzer", 
    analysis_instructions,
    ["data_analyzer"]
)
```

### Configuration-Driven Agent Tools

Use configuration to define agent hierarchies:

```python
import yaml
from typing import Any, Dict

class AgentToolFactory:
    """Factory for creating agent tools from configuration."""
    
    def __init__(self, agent_registry: Dict[str, Agent]):
        self.agent_registry = agent_registry
    
    def create_from_config(self, config: Dict[str, Any]) -> Agent:
        """Create orchestrator agent from configuration."""
        agent_name = config["name"]
        instructions = config["instructions"]
        
        tools = []
        for tool_config in config.get("tools", []):
            tool = self.create_tool_from_config(tool_config)
            tools.append(tool)
        
        return Agent(
            name=agent_name,
            instructions=lambda state: instructions,
            tools=tools
        )
    
    def create_tool_from_config(self, tool_config: Dict[str, Any]):
        """Create individual tool from configuration."""
        agent_name = tool_config["agent"]
        agent = self.agent_registry[agent_name]
        
        # Extract tool parameters
        params = {
            key: value for key, value in tool_config.items() 
            if key != "agent"
        }
        
        # Handle conditional enabling
        if "enabled_when" in params:
            condition = params.pop("enabled_when")
            params["is_enabled"] = self.create_condition(condition)
        
        return agent.as_tool(**params)
    
    def create_condition(self, condition: Dict[str, Any]):
        """Create enabling condition from configuration."""
        if condition["type"] == "context_attribute":
            return create_conditional_enabler(
                condition["attribute"],
                condition["value"]
            )
        # Add more condition types as needed
        return True

# Configuration file (config.yaml)
config_yaml = """
name: customer_service
instructions: "Route customers to appropriate specialists and handle their requests."

tools:
  - agent: technical_support
    tool_name: get_technical_help
    tool_description: "Get help with technical issues"
    max_turns: 10
    enabled_when:
      type: context_attribute
      attribute: request_type
      value: technical
  
  - agent: billing_support  
    tool_name: handle_billing
    tool_description: "Handle billing and payment issues"
    max_turns: 5
    enabled_when:
      type: context_attribute
      attribute: request_type
      value: billing
  
  - agent: general_support
    tool_name: general_assistance
    tool_description: "Provide general customer assistance"
    max_turns: 8
    preserve_session: true
"""

# Usage
config = yaml.safe_load(config_yaml)
factory = AgentToolFactory(agent_registry)
customer_service_agent = factory.create_from_config(config)
```

### Performance Optimization

Optimize agent tools for production:

```python
from functools import lru_cache
import asyncio

class OptimizedAgentTool:
    """Optimized agent tool with caching and pooling."""
    
    def __init__(self, agent: Agent, cache_size: int = 128):
        self.agent = agent
        self.cache_size = cache_size
        self.response_cache = {}
        self.execution_pool = asyncio.Semaphore(10)  # Limit concurrent executions
    
    @lru_cache(maxsize=128)
    def _cache_key(self, input_text: str, context_hash: str) -> str:
        """Generate cache key for responses."""
        return f"{input_text}:{context_hash}"
    
    async def execute_with_cache(self, input_text: str, context):
        """Execute with response caching."""
        # Generate context hash for cache key
        context_hash = str(hash(str(context)))
        cache_key = self._cache_key(input_text, context_hash)
        
        # Check cache first
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Limit concurrent executions
        async with self.execution_pool:
            # Double-check cache after acquiring semaphore
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Execute agent tool
            tool = self.agent.as_tool()
            result = await tool.execute({"input": input_text}, context)
            
            # Cache result
            self.response_cache[cache_key] = result
            
            # Cleanup cache if too large
            if len(self.response_cache) > self.cache_size:
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
            
            return result

# Usage with optimization
optimized_tool = OptimizedAgentTool(translator_agent, cache_size=256)
```

## Monitoring and Debugging

### Agent Tool Tracing

Monitor agent tool execution with detailed tracing:

```python
from jaf.core.tracing import ConsoleTraceCollector

def agent_tool_trace_handler(event):
    """Custom trace handler for agent tools."""
    if event.type == "run_start":
        data = event.data
        if "parent_run_id" in data:
            print(f"🔧 Agent tool started: {data.get('agent_name')} (parent: {data['parent_run_id']})")
    
    elif event.type == "run_end":
        data = event.data
        if "parent_run_id" in data:
            outcome = data.get("outcome", {})
            status = outcome.get("status", "unknown")
            print(f"✅ Agent tool completed: {status}")

# Enhanced tracing configuration
trace_collector = ConsoleTraceCollector()
composite_collector = create_composite_trace_collector(
    trace_collector,
    # Add custom handler for agent tools
    lambda event: agent_tool_trace_handler(event)
)

config = RunConfig(
    agent_registry=agents,
    model_provider=model_provider,
    on_event=composite_collector.collect
)
```

### Error Handling and Recovery

Implement robust error handling for agent tools:

```python
from jaf.core.agent_tool import create_default_output_extractor

def create_error_handling_extractor():
    """Create output extractor with error handling."""
    
    def error_extractor(run_result):
        try:
            if run_result.outcome.status == 'completed':
                return str(run_result.outcome.output)
            else:
                # Handle different error types
                error = run_result.outcome.error
                if hasattr(error, '_tag'):
                    error_type = error._tag
                    if error_type == "max_turns_exceeded":
                        return "Agent reached maximum turns. Partial result may be available."
                    elif error_type == "tool_timeout":
                        return "Agent execution timed out. Please try again."
                    elif error_type == "validation_error":
                        return "Input validation failed. Please check your request."
                
                return f"Agent execution failed: {str(error)}"
        
        except Exception as e:
            return f"Error processing agent result: {str(e)}"
    
    return error_extractor

# Create robust agent tools
robust_tool = agent.as_tool(
    custom_output_extractor=create_error_handling_extractor(),
    timeout=30.0,
    max_turns=5
)
```

### Testing Agent Tools

Test agent tools in isolation:

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_context():
    return Mock(
        user_id="test_user",
        permissions=["basic_access"],
        preferred_language="english"
    )

@pytest.fixture  
def test_agent():
    return Agent(
        name="test_agent",
        instructions=lambda state: "You are a test agent.",
        model_config=ModelConfig(name="gpt-4")
    )

async def test_agent_tool_creation(test_agent):
    """Test basic agent tool creation."""
    tool = test_agent.as_tool(
        tool_name="test_tool",
        tool_description="Test tool description"
    )
    
    assert tool.schema.name == "test_tool"
    assert "test tool description" in tool.schema.description.lower()

async def test_conditional_enabling(test_agent, mock_context):
    """Test conditional tool enabling."""
    def permission_check(context, agent):
        return "admin_access" in context.permissions
    
    tool = test_agent.as_tool(is_enabled=permission_check)
    
    # Tool should be disabled for basic user
    enabled = await tool._check_if_enabled(mock_context)
    assert not enabled
    
    # Tool should be enabled for admin user
    mock_context.permissions = ["admin_access"]
    enabled = await tool._check_if_enabled(mock_context)
    assert enabled

async def test_output_extraction(test_agent):
    """Test custom output extraction."""
    def extract_json(run_result):
        return '{"extracted": true}'
    
    tool = test_agent.as_tool(custom_output_extractor=extract_json)
    
    # Mock run result
    mock_result = Mock()
    mock_result.outcome.status = "completed"
    mock_result.outcome.output = "Some agent output"
    
    extracted = extract_json(mock_result)
    assert extracted == '{"extracted": true}'
```

## Best Practices

### Design Guidelines

1. **Single Responsibility**: Each agent tool should have a focused purpose
2. **Stateless Operations**: Prefer stateless agent tools when possible
3. **Clear Interfaces**: Use descriptive tool names and descriptions
4. **Error Handling**: Always handle agent tool failures gracefully
5. **Performance**: Monitor agent tool execution times and resource usage

### Configuration Management

1. **Environment-Based**: Use different tool configurations per environment
2. **Feature Flags**: Use conditional enabling for feature rollouts
3. **Version Control**: Version your agent tool configurations
4. **Documentation**: Document tool dependencies and requirements

### Security Considerations

1. **Permission Checks**: Validate user permissions before enabling tools
2. **Input Validation**: Sanitize inputs passed to agent tools
3. **Resource Limits**: Set appropriate timeouts and turn limits
4. **Audit Logging**: Log agent tool usage for security monitoring

### Scalability Patterns

1. **Tool Pooling**: Limit concurrent agent tool executions
2. **Caching**: Cache responses for idempotent operations
3. **Load Balancing**: Distribute agent tools across multiple instances
4. **Circuit Breakers**: Implement circuit breakers for failing agent tools

The agent-as-tool pattern in JAF enables sophisticated hierarchical agent architectures that are modular, maintainable, and scalable. By following these patterns and best practices, you can build complex multi-agent systems that leverage specialized expertise while maintaining clean separation of concerns.