# LLM Callbacks

LLM callbacks provide hooks to customize the behavior before and after LLM calls, enabling use cases like context summarization, logging, token optimization, and custom processing.

## Overview

JAF now supports two types of callbacks in `RunConfig`:

1. **`before_llm_call`**: Invoked before each LLM call, allowing you to modify the `RunState` (e.g., summarize messages, filter context)
2. **`after_llm_call`**: Invoked after each LLM call, allowing you to process or modify the `ModelCompletionResponse`

Both callbacks support synchronous and asynchronous functions.

## Type Signatures

```python
BeforeLLMCallback = Callable[
    [RunState[Ctx], Agent[Ctx, Any]],
    Union[RunState[Ctx], Awaitable[RunState[Ctx]]]
]

AfterLLMCallback = Callable[
    [RunState[Ctx], ModelCompletionResponse],
    Union[ModelCompletionResponse, Awaitable[ModelCompletionResponse]]
]
```

## Usage

### Basic Setup

Add callbacks to your `RunConfig`:

```python
from jaf.core.types import RunConfig

config = RunConfig(
    agent_registry=agents,
    model_provider=llm_service,
    before_llm_call=my_before_callback,  # Optional
    after_llm_call=my_after_callback,    # Optional
    # ... other config
)
```

### Example: Context Summarization

This is the primary use case - summarizing large contexts to reduce token usage:

```python
from dataclasses import replace
from jaf.core.types import RunState, Agent, Message, ContentRole

async def summarize_large_context(
    state: RunState,
    agent: Agent
) -> RunState:
    """Summarize messages if context is too large."""

    # Calculate token estimate
    total_chars = sum(len(str(msg.content)) for msg in state.messages)
    estimated_tokens = total_chars // 4

    # Threshold for summarization (e.g., 10K tokens)
    if estimated_tokens > 10000:
        # Keep recent messages, summarize older ones
        keep_recent = 5
        recent_messages = state.messages[-keep_recent:]
        old_messages = state.messages[:-keep_recent]

        # Create summary (you could use an LLM here for better summaries)
        summary = Message(
            role=ContentRole.ASSISTANT,
            content=f"[Previous conversation summarized: {len(old_messages)} messages]"
        )

        # Return modified state
        return replace(state, messages=[summary] + recent_messages)

    return state

# Use in config
config = RunConfig(
    agent_registry=agents,
    model_provider=llm_service,
    before_llm_call=summarize_large_context,
)
```

### Example: Advanced Summarization with LLM

For better quality summaries, you can use an LLM to summarize the old context:

```python
async def llm_based_summarization(
    state: RunState,
    agent: Agent
) -> RunState:
    """Use an LLM to create intelligent summaries."""

    total_chars = sum(len(str(msg.content)) for msg in state.messages)
    estimated_tokens = total_chars // 4

    if estimated_tokens > 10000 and len(state.messages) > 10:
        # Messages to summarize
        keep_recent = 5
        recent_messages = state.messages[-keep_recent:]
        old_messages = state.messages[:-keep_recent]

        # Create summarization prompt
        conversation_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in old_messages
        ])

        # Call LLM to summarize (pseudo-code)
        summary_response = await your_llm_service.summarize(
            text=conversation_text,
            max_length=200
        )

        summary_message = Message(
            role=ContentRole.ASSISTANT,
            content=f"[Context Summary]: {summary_response}"
        )

        return replace(state, messages=[summary_message] + recent_messages)

    return state
```

### Example: Logging and Monitoring

Track LLM interactions for debugging or analytics:

```python
import time
from jaf.core.types import ModelCompletionResponse

async def log_llm_interaction(
    state: RunState,
    response: ModelCompletionResponse
) -> ModelCompletionResponse:
    """Log LLM responses for monitoring."""

    # Log response details
    if response.message:
        print(f"[LLM Response] Content length: {len(response.message.content or '')}")

        if response.message.tool_calls:
            print(f"[LLM Response] Tool calls: {len(response.message.tool_calls)}")
            for tc in response.message.tool_calls:
                print(f"  - Tool: {tc.function.name}")

    # Could also send to analytics service
    # await analytics.track_llm_call(state, response)

    return response

config = RunConfig(
    agent_registry=agents,
    model_provider=llm_service,
    after_llm_call=log_llm_interaction,
)
```

### Example: Custom Token Counting

Track token usage across calls:

```python
class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.call_count = 0

    async def track_tokens(
        self,
        state: RunState,
        response: ModelCompletionResponse
    ) -> ModelCompletionResponse:
        """Track token usage."""
        self.call_count += 1

        # Estimate or extract actual token usage
        if hasattr(response, 'usage'):
            tokens = response.usage.get('total_tokens', 0)
            self.total_tokens += tokens
            print(f"[Tokens] Call #{self.call_count}: {tokens} tokens (Total: {self.total_tokens})")

        return response

tracker = TokenTracker()
config = RunConfig(
    agent_registry=agents,
    model_provider=llm_service,
    after_llm_call=tracker.track_tokens,
)
```

### Example: Response Filtering

Filter or modify LLM responses:

```python
async def filter_sensitive_content(
    state: RunState,
    response: ModelCompletionResponse
) -> ModelCompletionResponse:
    """Filter out sensitive information from responses."""

    if response.message and response.message.content:
        # Apply custom filtering logic
        filtered_content = response.message.content

        # Remove phone numbers, emails, etc.
        filtered_content = remove_phone_numbers(filtered_content)
        filtered_content = remove_emails(filtered_content)

        # Return modified response
        response.message.content = filtered_content

    return response

config = RunConfig(
    agent_registry=agents,
    model_provider=llm_service,
    after_llm_call=filter_sensitive_content,
)
```

## Combining Both Callbacks

You can use both callbacks together for comprehensive control:

```python
config = RunConfig(
    agent_registry=agents,
    model_provider=llm_service,
    before_llm_call=summarize_large_context,  # Optimize input
    after_llm_call=log_llm_interaction,       # Monitor output
)
```

## Best Practices

### 1. Keep Callbacks Fast

Callbacks are called on every LLM interaction. Keep them efficient:

```python
# ✅ Good - Quick check
async def quick_check(state: RunState, agent: Agent) -> RunState:
    if len(state.messages) > 100:
        return summarize(state)
    return state

# ❌ Bad - Expensive operation on every call
async def slow_check(state: RunState, agent: Agent) -> RunState:
    # Don't do heavy computation every time
    await expensive_database_query()
    await slow_external_api_call()
    return state
```

### 2. Handle Errors Gracefully

```python
async def safe_callback(state: RunState, agent: Agent) -> RunState:
    try:
        return await potentially_failing_operation(state)
    except Exception as e:
        # Log error but don't break the LLM call
        print(f"Callback error: {e}")
        return state  # Return unchanged state
```

### 3. Use Type Hints

```python
from jaf.core.types import RunState, Agent, ModelCompletionResponse

async def my_callback(
    state: RunState[MyContext],  # Specify your context type
    agent: Agent[MyContext, Any]
) -> RunState[MyContext]:
    # Type checking helps catch errors
    return state
```

### 4. Document Callback Behavior

```python
async def summarize_large_context(
    state: RunState,
    agent: Agent
) -> RunState:
    """
    Summarize conversation context when it exceeds token threshold.

    Behavior:
    - Triggers when estimated tokens > 10,000
    - Keeps last 5 messages intact
    - Replaces older messages with summary

    Args:
        state: Current run state with messages
        agent: Agent making the LLM call

    Returns:
        Modified state with summarized messages (or unchanged if below threshold)
    """
    # Implementation...
```

## Advanced Patterns

### Conditional Summarization

Different strategies based on agent or context:

```python
async def smart_summarization(
    state: RunState,
    agent: Agent
) -> RunState:
    # Different thresholds per agent
    thresholds = {
        "code_agent": 5000,   # Code agents need more context
        "chat_agent": 10000,  # Chat agents can handle more
    }

    threshold = thresholds.get(agent.name, 8000)
    total_chars = sum(len(str(msg.content)) for msg in state.messages)

    if total_chars // 4 > threshold:
        return await summarize(state, agent)

    return state
```

### Callback Chaining

Chain multiple callbacks:

```python
async def combined_before_callback(
    state: RunState,
    agent: Agent
) -> RunState:
    """Chain multiple operations."""
    state = await summarize_if_needed(state, agent)
    state = await filter_messages(state, agent)
    state = await add_custom_context(state, agent)
    return state
```

## See Also

- [Core Concepts](core-concepts.md)
- [RunConfig API Reference](api-reference.md#runconfig)
- [Example: llm_callback_example.py](../examples/llm_callback_example.py)
