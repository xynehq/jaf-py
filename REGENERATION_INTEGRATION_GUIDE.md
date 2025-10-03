# JAF Regeneration Integration with Existing Agents

This guide explains how JAF's regeneration system integrates with existing agents and provides practical examples for implementation.

## How Regeneration Works with Current Agents

### Core Integration Points

JAF regeneration is designed to be **agent-agnostic** and integrates seamlessly with existing agents through these key mechanisms:

#### 1. **Engine-Level Integration**
```python
# regeneration.py calls the normal engine after setting up truncated state
result = await engine_run(initial_state, regeneration_config)
```

The regeneration system:
- Loads the full conversation from memory
- Truncates messages at the specified regeneration point 
- Creates a new `RunState` with truncated conversation
- Passes control back to the normal `engine.run()` function
- **Your existing agent logic runs unchanged**

#### 2. **Memory Provider Integration**
```python
# Update memory with truncated conversation BEFORE running engine
store_result = await config.memory.provider.store_messages(
    conversation_id,
    truncated_messages,
    {**metadata, "regeneration_truncated": True}
)
```

#### 3. **Agent Registry Compatibility**
```python
# Uses the same agent registry and configuration
current_agent = config.agent_registry.get(state.current_agent_name)
```

## Practical Integration Examples

### Example 1: Basic Agent with Regeneration

```python
import asyncio
from jaf import Agent, run, RunConfig
from jaf.core.regeneration import regenerate_conversation
from jaf.core.types import RegenerationRequest, RunState, generate_run_id, generate_trace_id

# Your existing agent (unchanged)
async def search_web(query: str, context) -> str:
    return f"Search results for: {query}"

agent = Agent(
    name="search_agent",
    instructions="You help users search the web",
    tools=[search_web]
)

# Existing agent execution
async def run_normal_conversation():
    config = RunConfig(
        agent_registry={"search_agent": agent},
        conversation_id="conv_123",
        memory=memory_config  # Your memory setup
    )
    
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[{"role": "user", "content": "Search for Python tutorials"}],
        current_agent_name="search_agent",
        context={},
        turn_count=0
    )
    
    result = await run(initial_state, config)
    return result

# NEW: Regeneration capability (minimal code change)
async def regenerate_from_message():
    """Regenerate conversation from a specific message"""
    
    # Same config as normal execution
    config = RunConfig(
        agent_registry={"search_agent": agent},  # Same agent!
        conversation_id="conv_123",
        memory=memory_config
    )
    
    # Create regeneration request
    regen_request = RegenerationRequest(
        conversation_id="conv_123",
        message_id="msg_456",  # Message to regenerate from
        context={"regeneration_reason": "user_request"}
    )
    
    # Regenerate - agent logic runs normally after truncation
    result = await regenerate_conversation(
        regen_request,
        config,
        context={},
        agent_name="search_agent"
    )
    
    return result
```

### Example 2: Multi-Agent System with Regeneration

```python
# Your existing multi-agent setup
research_agent = Agent(
    name="researcher",
    instructions="You research topics thoroughly",
    tools=[search_web, analyze_data],
    handoffs=["writer"]  # Can handoff to writer
)

writer_agent = Agent(
    name="writer", 
    instructions="You write based on research",
    tools=[format_text, save_document]
)

# Normal multi-agent execution (unchanged)
async def run_research_pipeline():
    config = RunConfig(
        agent_registry={
            "researcher": research_agent,
            "writer": writer_agent
        },
        conversation_id="research_123",
        memory=memory_config
    )
    
    # Normal execution - agents can handoff as usual
    result = await run(initial_state, config)
    return result

# NEW: Regenerate from any point in multi-agent conversation
async def regenerate_research_step():
    """Regenerate from a specific point in multi-agent flow"""
    
    # Exact same config - all agents available
    config = RunConfig(
        agent_registry={
            "researcher": research_agent,  # Both agents still available
            "writer": writer_agent
        },
        conversation_id="research_123",
        memory=memory_config
    )
    
    regen_request = RegenerationRequest(
        conversation_id="research_123",
        message_id="researcher_msg_789",  # Regenerate from researcher's response
        context={"user_feedback": "Focus more on recent developments"}
    )
    
    # After regeneration, normal multi-agent flow continues
    # - Same handoff rules apply
    # - Same tool availability
    # - Same agent instructions
    result = await regenerate_conversation(
        regen_request,
        config,
        context={"user_feedback": "Focus more on recent developments"},
        agent_name="researcher"  # Resume from researcher
    )
    
    return result
```

### Example 3: Server Integration with Regeneration

```python
# Your existing JAF server setup
from jaf.server import create_server

# Current endpoints (unchanged)
app = create_server()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Your existing chat endpoint"""
    config = RunConfig(
        agent_registry={"assistant": assistant_agent},
        conversation_id=request.session_id,
        memory=memory_config
    )
    
    result = await run(initial_state, config)
    return {"response": result.outcome.output}

# NEW: Add regeneration endpoint (minimal addition)
@app.post("/regenerate") 
async def regenerate_endpoint(request: RegenerateRequest):
    """New regeneration endpoint using same agent config"""
    
    # Reuse EXACT same config as normal chat
    config = RunConfig(
        agent_registry={"assistant": assistant_agent},  # Same agent!
        conversation_id=request.session_id,
        memory=memory_config  # Same memory!
    )
    
    regen_request = RegenerationRequest(
        conversation_id=request.session_id,
        message_id=request.message_id,
        context=request.additional_context or {}
    )
    
    # Agent runs normally after regeneration setup
    result = await regenerate_conversation(
        regen_request,
        config,
        context=request.user_context,
        agent_name="assistant"
    )
    
    return {"regenerated_response": result.outcome.output}
```

### Example 4: HITL (Human-in-the-Loop) with Regeneration

```python
# Existing HITL setup with approval system
from jaf.core.types import ToolApprovalInterruption

approval_agent = Agent(
    name="approval_agent",
    instructions="You need approval for sensitive operations",
    tools=[sensitive_operation]  # Tool that requires approval
)

# Normal HITL flow (unchanged)
async def run_with_approvals():
    config = RunConfig(
        agent_registry={"approval_agent": approval_agent},
        conversation_id="approval_123",
        memory=memory_config,
        approval_storage=approval_storage  # Your approval system
    )
    
    result = await run(initial_state, config)
    
    if result.outcome.status == 'interrupted':
        # Handle approval interruption as usual
        handle_approval_request(result.outcome.interruptions)
    
    return result

# NEW: Regenerate after approval decisions
async def regenerate_with_different_approval():
    """Regenerate with different approval context"""
    
    # Same agent and config
    config = RunConfig(
        agent_registry={"approval_agent": approval_agent},
        conversation_id="approval_123", 
        memory=memory_config,
        approval_storage=approval_storage
    )
    
    # Regenerate from before the approval was needed
    regen_request = RegenerationRequest(
        conversation_id="approval_123",
        message_id="pre_approval_msg",
        context={
            "approval_context": "user_wants_alternative_approach",
            "user_preference": "avoid_sensitive_operations"
        }
    )
    
    # Agent will run again, potentially avoiding the approval scenario
    result = await regenerate_conversation(
        regen_request,
        config,
        context={"user_preference": "avoid_sensitive_operations"},
        agent_name="approval_agent"
    )
    
    return result
```

## Key Benefits for Existing Agents

### 1. **Zero Agent Code Changes**
- Your existing agent definitions remain unchanged
- Same instructions, tools, and handoff rules apply
- Same RunConfig structure

### 2. **Memory Consistency**  
- Regeneration updates the stored conversation
- Maintains full audit trail with regeneration points
- Original conversation history is preserved in metadata

### 3. **Tool and Approval Compatibility**
- All existing tools work normally after regeneration
- Approval systems continue to function
- Tool timeouts and error handling unchanged

### 4. **Multi-Agent Flow Preservation**
- Handoff rules remain active
- Agent capabilities unchanged
- Context passing works normally

## Integration Best Practices

### 1. **Conversation ID Management**
```python
# Use consistent conversation IDs
config = RunConfig(
    agent_registry=your_agents,
    conversation_id=session_id,  # Same for normal run and regeneration
    memory=memory_config
)
```

### 2. **Context Enrichment**
```python
# Add regeneration context for better outcomes
regen_request = RegenerationRequest(
    conversation_id=session_id,
    message_id=target_message_id,
    context={
        "user_feedback": "Be more concise",
        "regeneration_reason": "user_request",
        "iteration": 2
    }
)
```

### 3. **Error Handling**
```python
try:
    result = await regenerate_conversation(regen_request, config, context, agent_name)
    if result.outcome.status == 'error':
        # Handle regeneration-specific errors
        handle_regeneration_error(result.outcome.error)
except Exception as e:
    # Fallback to normal conversation flow
    result = await run(fallback_state, config)
```

### 4. **Memory Provider Requirements**
Ensure your memory provider supports:
- `store_messages()` - For updating truncated conversations
- `mark_regeneration_point()` - For audit trails (optional but recommended)
- `get_conversation()` - For loading conversation history

## Conclusion

JAF regeneration integrates seamlessly with existing agents by working at the conversation level rather than the agent level. Your agents continue to work exactly as before - regeneration simply provides a new starting point in the conversation history.

The integration requires minimal code changes and maintains full compatibility with:
- Multi-agent systems
- HITL workflows  
- Tool approval systems
- Memory providers
- Server deployments

This allows you to add powerful regeneration capabilities to your existing JAF applications without disrupting current functionality.
