# Parallel Agents API Reference

Complete API reference for JAF's parallel agent execution system.

## Module Import

```python
from jaf.core.parallel_agents import (
    ParallelAgentGroup,
    ParallelExecutionConfig,
    create_parallel_agents_tool,
    create_simple_parallel_tool,
    create_language_specialists_tool,
    create_domain_experts_tool
)
```

## Classes

### ParallelAgentGroup

Groups agents for parallel execution with shared configuration.

```python
@dataclass
class ParallelAgentGroup:
    name: str
    agents: List[Agent[Ctx, Out]]
    shared_input: bool = True
    result_aggregation: str = "combine"
    custom_aggregator: Optional[Callable[[List[str]], str]] = None
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Unique identifier for the group |
| `agents` | `List[Agent]` | Required | List of agents to execute in parallel |
| `shared_input` | `bool` | `True` | Whether all agents receive the same input |
| `result_aggregation` | `str` | `"combine"` | Strategy for combining results: `"combine"`, `"first"`, `"majority"`, `"custom"` |
| `custom_aggregator` | `Optional[Callable]` | `None` | Custom function for result aggregation (required if `result_aggregation="custom"`) |
| `timeout` | `Optional[float]` | `None` | Timeout in seconds for group execution |
| `metadata` | `Optional[Dict]` | `None` | Additional metadata for the group |

#### Example

```python
from jaf.core.parallel_agents import ParallelAgentGroup

group = ParallelAgentGroup(
    name="language_specialists",
    agents=[spanish_agent, french_agent, german_agent],
    shared_input=True,
    result_aggregation="combine",
    timeout=30.0,
    metadata={"category": "translation", "languages": 3}
)
```

### ParallelExecutionConfig

Configuration for executing multiple parallel agent groups.

```python
@dataclass
class ParallelExecutionConfig:
    groups: List[ParallelAgentGroup]
    inter_group_execution: str = "sequential"
    global_timeout: Optional[float] = None
    preserve_session: bool = False
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groups` | `List[ParallelAgentGroup]` | Required | List of parallel agent groups |
| `inter_group_execution` | `str` | `"sequential"` | How to execute groups: `"sequential"` or `"parallel"` |
| `global_timeout` | `Optional[float]` | `None` | Global timeout for all group executions |
| `preserve_session` | `bool` | `False` | Whether to preserve session across agent calls |

#### Example

```python
from jaf.core.parallel_agents import ParallelExecutionConfig

config = ParallelExecutionConfig(
    groups=[translation_group, analysis_group],
    inter_group_execution="parallel",
    global_timeout=120.0,
    preserve_session=True
)
```

### ParallelAgentsTool

Internal tool class that executes parallel agent groups. Usually created via convenience functions.

```python
class ParallelAgentsTool:
    def __init__(
        self,
        config: ParallelExecutionConfig,
        tool_name: str = "execute_parallel_agents",
        tool_description: str = "Execute multiple agents in parallel groups"
    )
    
    async def execute(self, args: AgentToolInput, context: Ctx) -> str
```

## Functions

### create_parallel_agents_tool()

Creates an advanced parallel agents tool with multiple groups and configuration options.

```python
def create_parallel_agents_tool(
    groups: List[ParallelAgentGroup],
    tool_name: str = "execute_parallel_agents",
    tool_description: str = "Execute multiple agents in parallel groups",
    inter_group_execution: str = "sequential",
    global_timeout: Optional[float] = None,
    preserve_session: bool = False
) -> Tool
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groups` | `List[ParallelAgentGroup]` | Required | List of parallel agent groups to execute |
| `tool_name` | `str` | `"execute_parallel_agents"` | Name of the created tool |
| `tool_description` | `str` | `"Execute multiple agents in parallel groups"` | Description of the tool |
| `inter_group_execution` | `str` | `"sequential"` | How to execute groups: `"sequential"` or `"parallel"` |
| `global_timeout` | `Optional[float]` | `None` | Global timeout for all executions |
| `preserve_session` | `bool` | `False` | Whether to preserve session across agent calls |

#### Returns

`Tool` - A JAF tool that can be used by agents

#### Example

```python
from jaf.core.parallel_agents import ParallelAgentGroup, create_parallel_agents_tool

# Create groups
rapid_response = ParallelAgentGroup(
    name="rapid_response",
    agents=[tech_agent, creative_agent],
    result_aggregation="first",
    timeout=15.0
)

analysis_team = ParallelAgentGroup(
    name="analysis_team",
    agents=[business_agent, legal_agent],
    result_aggregation="combine",
    timeout=30.0
)

# Create advanced tool
tool = create_parallel_agents_tool(
    groups=[rapid_response, analysis_team],
    tool_name="multi_team_consult",
    inter_group_execution="parallel",
    global_timeout=60.0
)
```

### create_simple_parallel_tool()

Creates a simple parallel tool from a list of agents with minimal configuration.

```python
def create_simple_parallel_tool(
    agents: List[Agent],
    group_name: str = "parallel_group",
    tool_name: str = "execute_parallel_agents",
    shared_input: bool = True,
    result_aggregation: str = "combine",
    timeout: Optional[float] = None
) -> Tool
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | Required | List of agents to execute in parallel |
| `group_name` | `str` | `"parallel_group"` | Name for the parallel group |
| `tool_name` | `str` | `"execute_parallel_agents"` | Name of the created tool |
| `shared_input` | `bool` | `True` | Whether all agents receive the same input |
| `result_aggregation` | `str` | `"combine"` | How to aggregate results |
| `timeout` | `Optional[float]` | `None` | Timeout for parallel execution |

#### Returns

`Tool` - A JAF tool that executes all agents in parallel

#### Example

```python
from jaf.core.parallel_agents import create_simple_parallel_tool

# Create simple parallel tool
tool = create_simple_parallel_tool(
    agents=[math_agent, science_agent, history_agent],
    group_name="expert_panel",
    tool_name="consult_experts",
    shared_input=True,
    result_aggregation="combine",
    timeout=30.0
)
```

### create_language_specialists_tool()

Creates a tool that consults multiple language specialists in parallel.

```python
def create_language_specialists_tool(
    language_agents: Dict[str, Agent],
    tool_name: str = "consult_language_specialists",
    timeout: Optional[float] = 30.0
) -> Tool
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language_agents` | `Dict[str, Agent]` | Required | Dictionary mapping language names to specialist agents |
| `tool_name` | `str` | `"consult_language_specialists"` | Name of the created tool |
| `timeout` | `Optional[float]` | `30.0` | Timeout for parallel execution |

#### Returns

`Tool` - A JAF tool for parallel language consultation

#### Example

```python
from jaf.core.parallel_agents import create_language_specialists_tool

# Create language specialists tool
tool = create_language_specialists_tool(
    language_agents={
        "spanish": spanish_agent,
        "french": french_agent,
        "german": german_agent,
        "italian": italian_agent
    },
    tool_name="translate_to_multiple_languages",
    timeout=25.0
)
```

### create_domain_experts_tool()

Creates a tool that consults multiple domain experts in parallel.

```python
def create_domain_experts_tool(
    expert_agents: Dict[str, Agent],
    tool_name: str = "consult_domain_experts",
    result_aggregation: str = "combine",
    timeout: Optional[float] = 60.0
) -> Tool
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expert_agents` | `Dict[str, Agent]` | Required | Dictionary mapping domain names to expert agents |
| `tool_name` | `str` | `"consult_domain_experts"` | Name of the created tool |
| `result_aggregation` | `str` | `"combine"` | How to aggregate expert results |
| `timeout` | `Optional[float]` | `60.0` | Timeout for parallel execution |

#### Returns

`Tool` - A JAF tool for parallel domain expert consultation

#### Example

```python
from jaf.core.parallel_agents import create_domain_experts_tool

# Create domain experts tool
tool = create_domain_experts_tool(
    expert_agents={
        "technology": tech_agent,
        "business": business_agent,
        "legal": legal_agent,
        "marketing": marketing_agent
    },
    tool_name="consult_advisory_board",
    result_aggregation="combine",
    timeout=45.0
)
```

## Result Aggregation

### Built-in Aggregation Strategies

#### "combine" (Default)

Combines all successful results into a structured format.

**Output Format:**
```python
{
    "combined_results": [
        "Result from agent 1",
        "Result from agent 2",
        "Result from agent 3"
    ],
    "result_count": 3
}
```

#### "first"

Returns the first successful result.

**Output Format:**
```python
"Result from first successful agent"
```

#### "majority"

Returns a result only if majority of agents succeed.

**Output Format (Success):**
```python
"Result from first agent (representing majority)"
```

**Output Format (No Majority):**
```python
{
    "error": "no_majority",
    "results": ["Result 1", "Result 2"],
    "message": "Only 2 out of 5 agents succeeded"
}
```

#### "custom"

Uses a custom aggregation function.

**Function Signature:**
```python
def custom_aggregator(results: List[str]) -> Union[str, Dict[str, Any]]:
    """
    Custom aggregation function.
    
    Args:
        results: List of successful result strings from agents
        
    Returns:
        Aggregated result (string or dict)
    """
    pass
```

**Example Custom Aggregator:**
```python
def consensus_aggregator(results):
    """Find consensus among results."""
    if len(results) < 2:
        return {"type": "single_result", "result": results[0] if results else "No results"}
    
    # Count common keywords
    keywords = {}
    for result in results:
        words = result.lower().split()
        for word in words:
            if len(word) > 4:  # Skip short words
                keywords[word] = keywords.get(word, 0) + 1
    
    # Find consensus themes
    consensus_words = {k: v for k, v in keywords.items() if v >= len(results) // 2}
    
    return {
        "type": "consensus_analysis",
        "agent_count": len(results),
        "consensus_keywords": list(consensus_words.keys()),
        "consensus_strength": len(consensus_words) / len(keywords) if keywords else 0,
        "all_results": results,
        "summary": f"Found {len(consensus_words)} consensus themes across {len(results)} agents"
    }

# Use in group
group = ParallelAgentGroup(
    name="consensus_team",
    agents=[agent1, agent2, agent3],
    result_aggregation="custom",
    custom_aggregator=consensus_aggregator
)
```

## Error Handling

### Common Error Scenarios

#### All Agents Failed

```python
{
    "error": "no_successful_results",
    "message": "All agents failed",
    "attempted_agents": 3,
    "failure_details": {
        "agent1": {"error": "timeout", "message": "Agent timed out after 30s"},
        "agent2": {"error": "execution_error", "message": "Invalid input format"},
        "agent3": {"error": "model_error", "message": "LLM provider error"}
    }
}
```

#### Partial Success

When some agents succeed and others fail, successful results are processed according to the aggregation strategy, and failed agents are noted in metadata.

#### Timeout Errors

```python
{
    "group_name": "analysis_team",
    "error": "timeout",
    "message": "Group analysis_team execution timed out after 45.0 seconds",
    "agent_count": 3,
    "completed_agents": 1  # How many completed before timeout
}
```

#### Custom Aggregation Errors

```python
{
    "error": "custom_aggregation_failed",
    "message": "Custom aggregator raised exception: division by zero",
    "successful_results": ["Result 1", "Result 2"],
    "aggregator_function": "consensus_aggregator"
}
```

## Integration Patterns

### With Existing JAF Features

#### Memory Integration

```python
# Agents can share memory when preserve_session=True
tool = create_simple_parallel_tool(
    agents=[agent1, agent2],
    preserve_session=True  # Agents share conversation memory
)
```

#### Conditional Tool Enabling

```python
from jaf.core.agent_tool import create_conditional_enabler

# Create conditional tools that work with parallel execution
conditional_tool = expert_agent.as_tool(
    tool_name="expert_analysis",
    is_enabled=create_conditional_enabler("priority", "high")
)

# Use in parallel group
group = ParallelAgentGroup(
    name="conditional_experts",
    agents=[expert_agent],  # Tool enabling happens at execution time
    result_aggregation="combine"
)
```

#### Tracing and Events

Parallel executions generate trace events for each agent:

```python
# Each parallel agent execution generates:
# - ToolCallStartEvent (when parallel group starts)
# - LLMCallStartEvent (for each agent)
# - LLMCallEndEvent (for each agent)
# - ToolCallEndEvent (when parallel group completes)
```

### Usage in Orchestrators

#### Single Parallel Tool

```python
orchestrator = Agent(
    name="simple_orchestrator",
    instructions=lambda state: "Use the parallel experts tool for comprehensive analysis",
    tools=[parallel_experts_tool]
)
```

#### Multiple Parallel Tools

```python
orchestrator = Agent(
    name="advanced_orchestrator",
    instructions=lambda state: '''You have access to multiple parallel tools:
    
    - consult_language_specialists: For translation and multilingual tasks
    - consult_domain_experts: For technical, business, and legal advice
    - analyze_data_parallel: For comprehensive data analysis
    
    Choose the appropriate tool(s) based on the user's request.
    You can even use multiple tools in one response for comprehensive help.''',
    tools=[language_tool, experts_tool, analysis_tool]
)
```

## Performance Optimization

### Timeout Strategy

```python
# Cascade timeouts: tool-specific < group < global
group = ParallelAgentGroup(
    name="optimized_group",
    agents=[
        quick_agent,    # Inherits group timeout (30s)
        slow_agent,     # Inherits group timeout (30s)
    ],
    timeout=30.0,       # Group timeout
)

tool = create_parallel_agents_tool(
    groups=[group],
    global_timeout=60.0  # Global timeout (higher than group)
)
```

### Batch Processing

For large numbers of agents, consider splitting into batches:

```python
def create_batched_parallel_tool(agents, batch_size=5, **kwargs):
    """Create parallel tool with batching for large agent lists."""
    if len(agents) <= batch_size:
        return create_simple_parallel_tool(agents, **kwargs)
    
    # Split into batches
    batches = [agents[i:i+batch_size] for i in range(0, len(agents), batch_size)]
    
    # Create groups for each batch
    groups = [
        ParallelAgentGroup(
            name=f"batch_{i}",
            agents=batch,
            result_aggregation=kwargs.get("result_aggregation", "combine"),
            timeout=kwargs.get("timeout")
        )
        for i, batch in enumerate(batches)
    ]
    
    return create_parallel_agents_tool(
        groups=groups,
        tool_name=kwargs.get("tool_name", "batched_parallel_agents"),
        inter_group_execution="sequential"  # Process batches sequentially
    )
```

## Migration Guide

### From Agent-as-Tool to Parallel Agents

**Before (Individual Tools):**
```python
spanish_tool = spanish_agent.as_tool(tool_name="translate_spanish")
french_tool = french_agent.as_tool(tool_name="translate_french")

orchestrator = Agent(
    name="translator",
    instructions="Use both translation tools",
    tools=[spanish_tool, french_tool]
)
```

**After (Parallel Agents):**
```python
translation_tool = create_language_specialists_tool(
    language_agents={"spanish": spanish_agent, "french": french_agent},
    tool_name="translate_parallel"
)

orchestrator = Agent(
    name="translator",
    instructions="Use the parallel translation tool",
    tools=[translation_tool]
)
```

### Benefits of Migration

1. **Guaranteed Parallelism**: Explicit parallel execution vs. LLM-dependent
2. **Result Aggregation**: Built-in strategies vs. manual synthesis
3. **Error Handling**: Graceful handling of partial failures
4. **Configuration**: Centralized timeout and execution settings
5. **Monitoring**: Better observability into parallel execution

This API reference provides complete documentation for all classes, functions, and patterns in JAF's parallel agent execution system.