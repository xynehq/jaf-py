# Parallel Agent Tools in JAF

JAF provides powerful capabilities for executing multiple agents as tools in parallel, enabling complex orchestration patterns and efficient resource utilization. This document covers the implementation and usage of parallel agent tools.

## Overview

JAF automatically executes multiple tool calls in parallel when an agent decides to call multiple tools in a single turn. The core engine uses `asyncio.gather()` to execute tool calls concurrently, providing automatic parallelism without additional configuration.

### Key Features

- **Automatic Parallel Execution**: Multiple tool calls in a single turn execute concurrently
- **Error Isolation**: Failures in one tool don't affect others
- **Timeout Management**: Individual tools can have their own timeout settings
- **Session Control**: Configure whether sub-agents share session state
- **Conditional Enabling**: Dynamic tool availability based on context
- **Hierarchical Patterns**: Multi-level agent orchestration

## Basic Parallel Execution

### Converting Agents to Tools

Any agent can be converted to a tool using the `as_tool()` method:

```python
from jaf.core.types import Agent
from jaf.core.config import ModelConfig

# Create specialized agents
data_validator = Agent(
    name="data_validator",
    instructions=lambda state: "Validate data quality and identify issues",
    model_config=ModelConfig(name="gpt-4", temperature=0.1)
)

stats_analyzer = Agent(
    name="stats_analyzer",
    instructions=lambda state: "Perform statistical analysis",
    model_config=ModelConfig(name="gpt-4", temperature=0.1)
)

# Convert to tools
validation_tool = data_validator.as_tool(
    tool_name="validate_data",
    tool_description="Validate data quality",
    max_turns=3,
    timeout=30.0
)

stats_tool = stats_analyzer.as_tool(
    tool_name="analyze_statistics",
    tool_description="Perform statistical analysis", 
    max_turns=3,
    timeout=45.0
)
```

### Creating Parallel Orchestrators

Create an orchestrator that uses multiple agent tools in parallel:

```python
orchestrator = Agent(
    name="data_analysis_orchestrator",
    instructions=lambda state: (
        "Analyze data using all available tools in parallel. "
        "Call validate_data and analyze_statistics simultaneously "
        "for comprehensive analysis."
    ),
    tools=[validation_tool, stats_tool],
    model_config=ModelConfig(name="gpt-4", temperature=0.1)
)
```

When the orchestrator decides to call both tools in a single response, JAF automatically executes them in parallel.

## Advanced Parallel Patterns

### 1. Conditional Parallel Execution

Enable tools conditionally based on context:

```python
from dataclasses import dataclass
from jaf.core.types import Context

@dataclass(frozen=True)
class AnalysisContext(Context):
    priority: str = "normal"
    user_type: str = "standard"

def high_priority_only(context: AnalysisContext, agent: Agent) -> bool:
    return context.priority == "high"

def premium_only(context: AnalysisContext, agent: Agent) -> bool:
    return context.user_type == "premium"

# Create conditional tools
quick_tool = quick_analyzer.as_tool(
    tool_name="quick_analysis",
    is_enabled=lambda ctx, agent: ctx.priority in ["normal", "high"]
)

deep_tool = deep_analyzer.as_tool(
    tool_name="deep_analysis", 
    is_enabled=high_priority_only
)

premium_tool = premium_analyzer.as_tool(
    tool_name="premium_analysis",
    is_enabled=premium_only
)
```

### 2. Hierarchical Parallel Execution

Create multi-level agent hierarchies with parallel execution at each level:

```python
# Level 3: Specialized processors
tokenizer = Agent(name="tokenizer", instructions=tokenizer_instructions)
entity_extractor = Agent(name="entity_extractor", instructions=entity_instructions)
sentiment_analyzer = Agent(name="sentiment_analyzer", instructions=sentiment_instructions)

# Level 2: Coordinator that uses Level 3 agents in parallel
text_processor = Agent(
    name="text_processor",
    instructions=lambda state: (
        "Coordinate text processing by using all analysis tools in parallel. "
        "Call tokenizer, entity_extractor, and sentiment_analyzer simultaneously."
    ),
    tools=[
        tokenizer.as_tool(tool_name="tokenize_text"),
        entity_extractor.as_tool(tool_name="extract_entities"),
        sentiment_analyzer.as_tool(tool_name="analyze_sentiment")
    ]
)

# Level 1: Main orchestrator
main_orchestrator = Agent(
    name="main_orchestrator", 
    instructions=lambda state: (
        "Coordinate comprehensive analysis using specialized processors in parallel."
    ),
    tools=[
        text_processor.as_tool(tool_name="process_text"),
        data_processor.as_tool(tool_name="process_data")
    ]
)
```

### 3. Session Management

Control how child agents inherit parent session state:

```python
# Ephemeral execution (default: preserve_session=False)
# Child agent gets fresh session, no shared memory
ephemeral_tool = agent.as_tool(preserve_session=False)

# Shared session (preserve_session=True)  
# Child agent shares parent's conversation_id and memory
shared_tool = agent.as_tool(preserve_session=True)
```

**Use Cases:**
- **Ephemeral**: Independent operations like translation, data validation
- **Shared**: Context-aware operations like customer service, personal assistance

## Utility Classes and Functions

### ParallelAgentRegistry

Centralized management of agents and their tool configurations:

```python
from jaf.core.parallel_tools import ParallelAgentRegistry

registry = ParallelAgentRegistry()

# Register agents with configurations
registry.register_agent(
    spanish_translator,
    tool_config={"tool_name": "translate_spanish", "max_turns": 3}
)

registry.register_agent(
    french_translator, 
    tool_config={"tool_name": "translate_french", "max_turns": 3}
)

# Create orchestrator from registered agents
translation_orchestrator = registry.create_parallel_orchestrator(
    name="translator",
    instructions="Translate text using available translation tools in parallel",
    agent_names=["spanish_translator", "french_translator"]
)
```

### ParallelToolsController

Advanced management with strategies and conditional execution:

```python
from jaf.core.parallel_tools import ParallelToolsController, ParallelExecutionConfig

controller = ParallelToolsController()

# Register agents
controller.registry.register_agent(analyzer1)
controller.registry.register_agent(analyzer2)

# Register conditions
controller.conditional_manager.register_condition(
    "high_priority",
    lambda ctx: getattr(ctx, 'priority', 'normal') == 'high'
)

# Create adaptive orchestrator
orchestrator = controller.create_adaptive_orchestrator(
    name="adaptive_analyzer",
    instructions="Analyze data using appropriate tools based on context",
    agent_configs=[
        {
            'agent_name': 'analyzer1',
            'condition': 'high_priority',
            'tool_config': {'timeout': 60.0}
        },
        {
            'agent_name': 'analyzer2', 
            'condition': True,  # Always enabled
            'tool_config': {'timeout': 30.0}
        }
    ]
)

# Execute with strategy
config = ParallelExecutionConfig(
    max_concurrent=5,
    timeout_per_tool=45.0,
    batch_size=3
)

result = await controller.execute_with_strategy(
    orchestrator=orchestrator,
    context=analysis_context,
    message="Analyze this data",
    strategy='batched',
    config=config
)
```

## Convenience Functions

### Analysis Pipeline

Create a parallel analysis pipeline from multiple analyzers:

```python
from jaf.core.parallel_tools import create_analysis_pipeline

# Create analyzers
analyzers = [data_validator, stats_analyzer, viz_recommender, business_analyst]

# Create pipeline that runs all analyzers in parallel
pipeline = create_analysis_pipeline(
    analyzers=analyzers,
    orchestrator_name="comprehensive_analyzer",
    max_turns=3,
    timeout=60.0
)
```

### Conditional Processor

Create a processor with context-based tool enabling:

```python
from jaf.core.parallel_tools import create_conditional_processor

processors = {
    'quick': quick_processor,
    'detailed': detailed_processor,
    'premium': premium_processor
}

conditions = {
    'quick': lambda ctx: ctx.priority == 'normal',
    'detailed': lambda ctx: ctx.priority == 'high', 
    'premium': lambda ctx: ctx.user_type == 'premium'
}

conditional_processor = create_conditional_processor(
    processors=processors,
    conditions=conditions,
    orchestrator_name="smart_processor"
)
```

## Best Practices

### 1. Design for Parallelism

- **Independent Tools**: Design agent tools to be independent and stateless when possible
- **Clear Interfaces**: Use well-defined input/output contracts between agents
- **Error Handling**: Implement proper error handling in each agent

### 2. Resource Management

- **Timeouts**: Set appropriate timeouts for each tool based on expected execution time
- **Batch Processing**: Use batching for large numbers of agents to control resource usage
- **Monitoring**: Monitor parallel execution performance and adjust configurations

### 3. Context Design

- **Shared State**: Use context to share necessary state between parallel agents
- **Conditional Logic**: Implement context-based conditional enabling for dynamic behavior
- **Type Safety**: Use dataclasses for strongly-typed context objects

### 4. Orchestration Patterns

- **Clear Instructions**: Provide clear instructions about when to use tools in parallel
- **Result Synthesis**: Design orchestrators to effectively combine parallel results
- **Fallback Strategies**: Implement fallback behavior when some tools fail

## Examples

### Language Specialists Demo

A complete example demonstrating parallel language agents working together:

```python
"""
JAF Parallel Language Agents Demo

This example demonstrates how to use multiple language-specific agents as tools that execute in parallel
when called simultaneously by an orchestrator agent. JAF automatically handles
parallel execution of multiple tool calls within a single turn.
"""

import asyncio
import os
from dataclasses import dataclass
from jaf import Agent, make_litellm_provider
from jaf.core.types import RunState, RunConfig, Message, generate_run_id, generate_trace_id
from jaf.core.engine import run

def setup_litellm_provider():
    """Setup LiteLLM provider using environment variables."""
    base_url = os.getenv('LITELLM_BASE_URL', 'https://grid.ai.juspay.net/')
    api_key = os.getenv('LITELLM_API_KEY')
    
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is required")
    
    return make_litellm_provider(base_url, api_key)

# Create specialized language agents
def create_language_agents():
    """Create German and French language agents."""
    
    # German Agent
    german_agent = Agent(
        name='german_specialist',
        instructions=lambda state: '''Du bist ein deutscher Sprachspezialist. 
        
Deine Aufgaben:
- Antworte IMMER auf Deutsch
- Übersetze gegebene Texte ins Deutsche
- Erkläre deutsche Kultur und Sprache
- Sei freundlich und hilfsbereit
- Verwende authentische deutsche Ausdrücke
        
Du hilfst Menschen dabei, deutsche Sprache und Kultur zu verstehen.''',
        tools=[]
    )
    
    # French Agent  
    french_agent = Agent(
        name='french_specialist',
        instructions=lambda state: '''Tu es un spécialiste de la langue française.
        
Tes tâches:
- Réponds TOUJOURS en français
- Traduis les textes donnés en français
- Explique la culture et la langue françaises
- Sois aimable et serviable
- Utilise des expressions françaises authentiques
        
Tu aides les gens à comprendre la langue et la culture françaises.''',
        tools=[]
    )
    
    return german_agent, french_agent

# Create the language agents
german_agent, french_agent = create_language_agents()

# Convert agents to tools using the as_tool() method
german_tool = german_agent.as_tool(
    tool_name='ask_german_specialist',
    tool_description='Ask the German language specialist to respond in German or translate to German'
)

french_tool = french_agent.as_tool(
    tool_name='ask_french_specialist', 
    tool_description='Ask the French language specialist to respond in French or translate to French'
)

@dataclass
class LanguageContext:
    """Context for language operations."""
    user_id: str = "demo_user"
    request_id: str = "lang_demo_001"
    languages: list = None
    task_type: str = "translation"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["german", "french"]

async def demo_parallel_execution():
    """Demonstrate parallel language agent execution."""
    
    # Setup model provider
    model_provider = setup_litellm_provider()
    
    # Create orchestrator agent with parallel language tools
    orchestrator = Agent(
        name='language_orchestrator',
        instructions=lambda state: '''You are a language orchestrator that coordinates multiple language specialists.

When given a message to translate or respond to:

1. Call BOTH language specialists in parallel in the same response
2. Use ask_german_specialist and ask_french_specialist simultaneously 
3. After receiving both responses, provide a summary comparing the responses
4. Be helpful and explain any cultural nuances between the languages

IMPORTANT: Always call both language tools in the same response to demonstrate parallel execution.''',
        tools=[german_tool, french_tool]
    )
    
    # Create context
    context = LanguageContext(
        user_id="demo_user",
        request_id="lang_demo_001",
        languages=["german", "french"],
        task_type="multilingual_response"
    )
    
    # Test message
    test_message = "Hello! How are you doing today? I hope you are having a wonderful time learning new languages!"
    
    # Create agent registry with all agents
    agent_registry = {
        'language_orchestrator': orchestrator,
        'german_specialist': german_agent,
        'french_specialist': french_agent
    }
    
    # Create run state
    run_id = generate_run_id()
    trace_id = generate_trace_id()
    
    initial_state = RunState(
        run_id=run_id,
        trace_id=trace_id,
        messages=[Message(role='user', content=f"Please have both language specialists respond to this message in parallel: {test_message}")],
        current_agent_name='language_orchestrator',
        context=context.__dict__,
        turn_count=0
    )
    
    # Create run config
    config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=3,
        model_override='gemini-2.5-pro'
    )
    
    print("Starting JAF execution...")
    print("WATCH: JAF will automatically detect multiple tool calls and execute them in parallel")
    
    # Execute with timing
    import time
    start_time = time.time()
    
    result = await run(initial_state, config)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution completed in {execution_time:.2f} seconds")
    print(f"Final Status: {result.outcome.status}")
    
    # Show the conversation flow
    for message in result.final_state.messages:
        if message.role == 'user':
            print(f"\nUser Request:")
            print(message.content)
        elif message.role == 'assistant':
            print(f"\nAssistant Response:")
            print(message.content)
            print("-" * 40)

if __name__ == "__main__":
    asyncio.run(demo_parallel_execution())
```

**Key Features Demonstrated:**

1. **Agent-as-Tool Pattern**: Language agents converted to tools using `.as_tool()`
2. **Parallel Execution**: JAF automatically executes both language tools simultaneously
3. **Cultural Context**: Each agent maintains its authentic language and cultural context
4. **Orchestration**: Main agent coordinates and synthesizes parallel responses
5. **Performance**: Parallel execution reduces response time compared to sequential calls

**Expected Output:**
- German specialist responds in authentic German
- French specialist responds in authentic French  
- Both responses generated concurrently
- Orchestrator provides comparative analysis

### Multi-Modal Analysis

```python
# Analyze different data types in parallel
text_analyzer = Agent(name="text_analyzer", instructions=text_instructions)
image_analyzer = Agent(name="image_analyzer", instructions=image_instructions)  
audio_analyzer = Agent(name="audio_analyzer", instructions=audio_instructions)

multimodal_orchestrator = Agent(
    name="multimodal_analyzer",
    instructions=lambda state: (
        "Analyze the provided content using appropriate analyzers in parallel. "
        "Use text_analyzer for text content, image_analyzer for images, "
        "and audio_analyzer for audio files. Combine results into unified analysis."
    ),
    tools=[
        text_analyzer.as_tool(tool_name="analyze_text"),
        image_analyzer.as_tool(tool_name="analyze_image"),
        audio_analyzer.as_tool(tool_name="analyze_audio")
    ]
)
```

### Customer Service Pipeline

```python
# Handle customer requests with parallel specialist routing
technical_support = Agent(name="technical_support", instructions=tech_instructions)
billing_support = Agent(name="billing_support", instructions=billing_instructions)
general_support = Agent(name="general_support", instructions=general_instructions)

def technical_enabled(context, agent):
    return "technical" in context.request_type.lower()

def billing_enabled(context, agent):
    return "billing" in context.request_type.lower()

customer_service = Agent(
    name="customer_service",
    instructions=lambda state: (
        "Route customer requests to appropriate specialists. "
        "Use multiple specialists in parallel when request spans multiple areas."
    ),
    tools=[
        technical_support.as_tool(
            tool_name="get_technical_help",
            is_enabled=technical_enabled
        ),
        billing_support.as_tool(
            tool_name="handle_billing",
            is_enabled=billing_enabled
        ),
        general_support.as_tool(
            tool_name="general_assistance",
            preserve_session=True  # Keep conversation context
        )
    ]
)
```

## Performance Considerations

### Concurrency Control

- JAF uses `asyncio.gather()` for parallel execution
- Each tool call is executed independently with proper error isolation
- Failed tools don't block other parallel executions

### Memory Usage

- Consider memory usage when running many agents in parallel
- Use ephemeral sessions for independent operations to reduce memory overhead
- Implement cleanup for long-running orchestrators

### Timeout Management

- Set realistic timeouts based on tool complexity
- Use shorter timeouts for quick operations, longer for complex analysis
- Implement timeout handling in orchestrator instructions

## Integration with JAF Features

### Memory Integration

Parallel agent tools work seamlessly with JAF's memory system:

```python
# Tools can access shared memory
analyzer_tool = analyzer.as_tool(
    preserve_session=True,  # Share memory with parent
    tool_name="analyze_with_memory"
)
```

### Tracing and Monitoring

Parallel executions are fully traced in JAF's monitoring system:

- Each tool call gets individual trace events
- Parallel execution timing is tracked
- Errors in individual tools are isolated and reported

### Function Composition

Combine with JAF's function composition for advanced patterns:

```python
from jaf.core.composition import compose_functions

# Compose parallel analysis with post-processing
composed_analyzer = compose_functions([
    parallel_orchestrator,
    result_synthesizer,
    report_generator
])
```

## Forced Parallel Execution

### Overview

While JAF automatically executes multiple tool calls in parallel when an LLM decides to call them together, sometimes you need **guaranteed** parallel execution regardless of the LLM's decision-making. The forced parallel execution feature ensures deterministic parallel behavior.

### Why Force Parallel Execution?

- **Consistency**: Ensure all tools execute every time, regardless of LLM decisions
- **Performance**: Guarantee maximum parallelism for time-critical operations
- **Comprehensive Analysis**: Force complete analysis from all available perspectives
- **Deterministic Behavior**: Predictable execution patterns for production systems

### Forced Parallel Modes

```python
from jaf.core.forced_parallel import ParallelMode, ForceParallelConfig

class ParallelMode(Enum):
    AUTO = "auto"                    # Normal JAF behavior (LLM decides)
    FORCE_ALL = "force_all"          # Force ALL tools to execute
    FORCE_GROUPS = "force_groups"    # Force specific groups in parallel
    CONDITIONAL_FORCE = "conditional_force"  # Force based on conditions
```

### Force All Tools Pattern

Force execution of ALL available tools regardless of LLM decision:

```python
from jaf.core.forced_parallel import create_force_all_orchestrator

# Create specialized agents
data_validator = Agent(name="validator", instructions=validation_instructions)
stats_analyzer = Agent(name="analyzer", instructions=analysis_instructions)
viz_recommender = Agent(name="visualizer", instructions=viz_instructions)
business_analyst = Agent(name="business", instructions=business_instructions)

# Create orchestrator that ALWAYS uses ALL tools in parallel
force_all_orchestrator = create_force_all_orchestrator(
    name="comprehensive_analyzer",
    agents=[data_validator, stats_analyzer, viz_recommender, business_analyst],
    timeout_per_tool=45.0
)

# This will ALWAYS execute all 4 tools in parallel, no matter what
result = await run_agent(force_all_orchestrator, ["Analyze this data"], context)
```

### Grouped Forced Execution

Force execution of specific tool groups based on conditions:

```python
from jaf.core.forced_parallel import ParallelGroup, create_grouped_parallel_orchestrator

# Define tool groups
data_analysis_group = ParallelGroup(
    name="data_analysis",
    tool_names=["execute_validator", "execute_analyzer"],
    condition=lambda ctx, msg: "data" in msg.lower(),
    description="Core data analysis tools"
)

business_group = ParallelGroup(
    name="business_insights",
    tool_names=["execute_visualizer", "execute_business"],
    condition=lambda ctx, msg: "business" in msg.lower(),
    description="Business intelligence tools"
)

# Create grouped orchestrator
grouped_orchestrator = create_grouped_parallel_orchestrator(
    name="grouped_analyzer",
    agents=[data_validator, stats_analyzer, viz_recommender, business_analyst],
    groups=[data_analysis_group, business_group],
    timeout_per_tool=40.0
)
```

### Conditional Forced Execution

Force parallel execution based on runtime conditions:

```python
from jaf.core.forced_parallel import (
    create_conditional_parallel_orchestrator,
    create_priority_condition,
    create_keyword_condition,
    create_user_type_condition
)

# Force parallel execution for high-priority requests
priority_condition = create_priority_condition("high")

conditional_orchestrator = create_conditional_parallel_orchestrator(
    name="conditional_analyzer", 
    agents=[validator, analyzer, business_analyst],
    force_condition=priority_condition,
    timeout_per_tool=60.0
)

# Force parallel execution based on keywords
keyword_condition = create_keyword_condition(["urgent", "comprehensive", "complete"])

keyword_orchestrator = create_conditional_parallel_orchestrator(
    name="keyword_triggered",
    agents=[validator, analyzer],
    force_condition=keyword_condition
)

# Force parallel execution for premium users
user_condition = create_user_type_condition(["premium", "enterprise"])

premium_orchestrator = create_conditional_parallel_orchestrator(
    name="premium_analyzer",
    agents=[validator, analyzer, premium_analyst],
    force_condition=user_condition
)
```

### Custom Forced Parallel Agent

Wrap any existing agent with forced parallel behavior:

```python
from jaf.core.forced_parallel import ForcedParallelAgent, ForceParallelConfig, ParallelMode

# Create base orchestrator
base_orchestrator = Agent(
    name="base_analyzer",
    instructions="Analyze data using available tools",
    tools=[
        validator.as_tool(tool_name="validate"),
        analyzer.as_tool(tool_name="analyze"),
        reporter.as_tool(tool_name="report")
    ]
)

# Configure forced parallel execution
force_config = ForceParallelConfig(
    mode=ParallelMode.FORCE_ALL,
    timeout_per_tool=40.0,
    max_retries=1,
    fallback_to_auto=True
)

# Wrap with forced parallel behavior
forced_agent = ForcedParallelAgent(
    agent=base_orchestrator,
    config=force_config,
    name_suffix="_forced"
)

# Now ALL tools will execute in parallel every time
result = await forced_agent.run(["Analyze this"], context, run_config)
```

### Registry-Based Forced Orchestration

Use the registry for centralized forced parallel management:

```python
from jaf.core.parallel_tools import ParallelAgentRegistry

registry = ParallelAgentRegistry()

# Register agents with configurations
registry.register_agent(quick_analyzer, {"timeout": 15.0}, groups=["basic"])
registry.register_agent(deep_analyzer, {"timeout": 45.0}, groups=["advanced"])
registry.register_agent(premium_analyzer, {"timeout": 60.0}, groups=["premium"])

# Create forced parallel orchestrator from registry
force_all = registry.create_forced_parallel_orchestrator(
    name="registry_force_all",
    instructions="Execute ALL registered agents in parallel",
    force_mode="force_all",
    timeout=30.0
)

# Create group-specific forced orchestrator
advanced_only = registry.create_forced_parallel_orchestrator(
    name="advanced_forced",
    instructions="Execute advanced agents in parallel",
    agent_names=["deep_analyzer", "premium_analyzer"],
    force_mode="force_all"
)
```

### Performance Benefits

**Sequential Execution (Traditional):**
```
Tool 1 → (60s) → Tool 2 → (60s) → Tool 3 → (60s) → Tool 4
Total Time: 240 seconds
```

**Forced Parallel Execution:**
```
Tool 1 ┐
Tool 2 ├─ All execute simultaneously
Tool 3 ├─ All execute simultaneously  
Tool 4 ┘
Total Time: 60 seconds (75% time reduction)
```

### Advanced Patterns

#### Multi-Modal Forced Analysis

```python
# Force different analysis types in parallel
text_analyzer = create_force_all_orchestrator(
    name="text_analysis",
    agents=[sentiment_analyzer, entity_extractor, summarizer]
)

data_analyzer = create_force_all_orchestrator(
    name="data_analysis", 
    agents=[validator, stats_analyzer, viz_recommender]
)

# Hierarchical forced execution
main_orchestrator = Agent(
    name="multi_modal",
    instructions="Perform comprehensive multi-modal analysis",
    tools=[
        text_analyzer.as_tool(tool_name="analyze_text"),
        data_analyzer.as_tool(tool_name="analyze_data")
    ]
)
```

#### Context-Aware Forced Execution

```python
def smart_force_condition(context, message):
    """Smart condition that considers multiple factors."""
    is_urgent = "urgent" in message.lower()
    is_high_priority = getattr(context, 'priority', 'normal') == 'high'
    is_premium = getattr(context, 'user_type', 'standard') == 'premium'
    
    return is_urgent or (is_high_priority and is_premium)

smart_orchestrator = create_conditional_parallel_orchestrator(
    name="smart_forced",
    agents=[validator, analyzer, business_analyst, premium_specialist],
    force_condition=smart_force_condition
)
```

### Best Practices for Forced Parallel Execution

1. **Choose the Right Mode**:
   - Use `FORCE_ALL` for comprehensive analysis
   - Use `FORCE_GROUPS` for context-specific tool sets
   - Use `CONDITIONAL_FORCE` for dynamic behavior

2. **Set Appropriate Timeouts**:
   - Consider the slowest tool in the parallel set
   - Allow extra time for parallel coordination overhead
   - Use different timeouts for different tool types

3. **Handle Failures Gracefully**:
   - Enable `fallback_to_auto` for production reliability
   - Monitor individual tool success rates
   - Implement retry logic for critical operations

4. **Optimize Instructions**:
   - Make it clear that ALL tools should be used
   - Explain the parallel execution expectation
   - Provide context for result synthesis

5. **Monitor Performance**:
   - Track parallel execution times
   - Monitor resource utilization
   - Measure improvement over sequential execution

### Common Use Cases

- **Comprehensive Data Analysis**: Force all analysis tools for complete insights
- **Multi-Perspective Reviews**: Force all reviewers for thorough evaluation  
- **Time-Critical Operations**: Force parallelism for maximum speed
- **Quality Assurance**: Force all validation tools for complete coverage
- **Research Analysis**: Force all research tools for comprehensive findings

This comprehensive guide covers all aspects of parallel agent tools in JAF, from basic usage to advanced forced parallel patterns and best practices.