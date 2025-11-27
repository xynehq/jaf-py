# Langfuse Cost Tracking Fix - Implementation Summary

## Overview
Fixed Langfuse cost tracking showing $0.00 when using LiteLLM SDK streaming provider with Azure OpenAI models. The issue occurred because OpenAI's streaming API sends usage data in a separate chunk after the finish chunk, which wasn't being captured by JAF's streaming implementation.

## Problem Statement
When using JAF with LiteLLM streaming provider and Azure OpenAI (e.g., `azure/gpt-4.1`), Langfuse traces showed:
- **Cost**: $0.00 (incorrect)
- **Token counts**: 0 prompt tokens, 0 completion tokens (missing)
- **Model**: Correctly identified (e.g., `gpt-4.1-2025-04-14`)

## Root Cause
1. OpenAI streaming with `stream_options: {"include_usage": true}` sends usage data in a **final, separate chunk** after the finish chunk
2. JAF's `model.py` was not yielding this usage-only chunk
3. JAF's `engine.py` was not extracting usage/model from streaming chunks
4. Without usage data, Langfuse cannot calculate costs

## Solution
The fix involved changes to **two files** in the JAF package to capture and propagate usage/model data through the streaming pipeline.

---

## Files Modified

### 1. `jaf/providers/model.py`
**Function**: `make_litellm_sdk_provider()` → `get_completion_stream()`

#### Change 1: Request usage data in streaming
```python
# Request usage data in streaming chunks
request_params["stream_options"] = {"include_usage": True}
```
- **Location**: Before `await litellm.acompletion(**request_params)`
- **Purpose**: Tells OpenAI to include usage data in streaming responses

#### Change 2: Initialize tracking variables
```python
# Variables to accumulate usage and model from streaming chunks
accumulated_usage = None
response_model = None
```
- **Location**: Right after `stream = await litellm.acompletion(**request_params)`
- **Purpose**: Track usage and model data across all chunks

#### Change 3: Extract usage and model from chunks
```python
try:
    raw_obj = chunk.model_dump() if hasattr(chunk, "model_dump") else None
    
    # Capture usage from chunk if present
    if raw_obj and "usage" in raw_obj and raw_obj["usage"]:
        accumulated_usage = raw_obj["usage"]
    
    # Capture model from chunk if present
    if raw_obj and "model" in raw_obj and raw_obj["model"]:
        response_model = raw_obj["model"]
        
except Exception:
    raw_obj = None
```
- **Location**: In the chunk extraction try/except block
- **Purpose**: Capture usage and model data as soon as they appear in any chunk

#### Change 4: Yield usage-only chunks
```python
# Handle usage-only chunks (OpenAI streaming with stream_options includes usage in final chunk)
# These chunks have usage but may have empty or no choice/delta
if raw_obj and "usage" in raw_obj and raw_obj["usage"]:
    # Yield this chunk so engine.py can capture usage from raw
    yield CompletionStreamChunk(delta="", raw=raw_obj)
```
- **Location**: After raw_obj extraction, before choice processing
- **Purpose**: Ensure usage-only chunks are sent to engine.py for processing

#### Change 5: Include usage/model in content delta chunks
```python
if content_delta:
    # Include accumulated usage and model in raw_obj for engine
    if raw_obj and (accumulated_usage or response_model):
        if accumulated_usage:
            raw_obj["usage"] = accumulated_usage
        if response_model:
            raw_obj["model"] = response_model
    yield CompletionStreamChunk(delta=content_delta, raw=raw_obj)
```
- **Location**: In content delta handling
- **Purpose**: Propagate usage/model data with content chunks

#### Change 6: Include usage/model in tool call delta chunks
```python
# Include accumulated usage and model in raw_obj
if raw_obj and (accumulated_usage or response_model):
    if accumulated_usage:
        raw_obj["usage"] = accumulated_usage
    if response_model:
        raw_obj["model"] = response_model

yield CompletionStreamChunk(
    tool_call_delta=ToolCallDelta(...),
    raw=raw_obj,
)
```
- **Location**: Inside tool call delta loop
- **Purpose**: Propagate usage/model data with tool call chunks

#### Change 7: Include usage/model in finish chunk
```python
if finish_reason:
    # Include accumulated usage and model in final chunk
    if raw_obj and (accumulated_usage or response_model):
        if accumulated_usage:
            raw_obj["usage"] = accumulated_usage
        if response_model:
            raw_obj["model"] = response_model
    yield CompletionStreamChunk(
        is_done=True, finish_reason=finish_reason, raw=raw_obj
    )
```
- **Location**: In finish_reason handling
- **Purpose**: Ensure final chunk has complete usage/model data

### 2. `jaf/core/engine.py`
**Function**: `_agent_turn()` → streaming block

#### Change 1: Initialize tracking variables
```python
if use_streaming:
    try:
        aggregated_text = ""
        partial_tool_calls: List[Dict[str, Any]] = []
        # Capture usage and model from streaming chunks
        stream_usage: Optional[Dict[str, int]] = None
        stream_model: Optional[str] = None
```
- **Location**: At the start of the streaming try block
- **Purpose**: Track usage and model data extracted from chunks

#### Change 2: Extract usage and model from chunks
```python
async for chunk in get_stream(state, current_agent, config):
    # Extract usage and model from raw chunk if available
    raw_chunk = getattr(chunk, "raw", None)
    if raw_chunk:
        if not stream_usage and "usage" in raw_chunk and raw_chunk["usage"]:
            stream_usage = raw_chunk["usage"]
        if not stream_model and "model" in raw_chunk and raw_chunk["model"]:
            stream_model = raw_chunk["model"]
    
    # Text deltas
    delta_text = getattr(chunk, "delta", None)
```
- **Location**: At the start of the chunk processing loop
- **Purpose**: Extract and store usage/model data from each chunk's raw field

#### Change 3: Add usage/model to llm_response
```python
llm_response = {
    "message": {"content": aggregated_text or None, "tool_calls": final_tool_calls}
}

# Preserve usage and model from streaming if captured
if stream_usage:
    llm_response["usage"] = stream_usage
if stream_model:
    llm_response["model"] = stream_model
```
- **Location**: After building llm_response, before the except block
- **Purpose**: Include captured usage and model in final response sent to Langfuse

---

## Backward Compatibility

### ✅ Fully Backward Compatible - No Breaking Changes

1. **No data structure changes**: `CompletionStreamChunk` has all optional fields
2. **Additive only**: Only adding optional `usage` and `model` fields to `llm_response`
3. **Consistent with existing API**: Non-streaming `get_completion()` already returns these fields
4. **Safe extraction**: All new code uses defensive conditionals
5. **No signature changes**: No parameters added/removed from any functions
6. **Graceful degradation**: If usage data isn't available, code continues without errors

## Testing Verification

After applying these changes, verify with:

```python
from jaf import Agent, make_litellm_sdk_provider

agent = Agent(
    name="test_agent",
    model=make_litellm_sdk_provider(model="azure/gpt-4.1"),
    instructions="You are a helpful assistant."
)

result = agent.run("Hello, how are you?")
```

**Expected Results in Langfuse:**
- ✅ Non-zero cost (e.g., $0.04268)
- ✅ Correct token counts (prompt_tokens, completion_tokens, total_tokens)
- ✅ Correct model ID (e.g., `gpt-4.1-2025-04-14`)

## Impact

- ✅ Cost tracking works correctly for all LiteLLM streaming providers
- ✅ Token usage accurately reported to Langfuse
- ✅ Works with both text and tool call responses
- ✅ No breaking changes to existing API
- ✅ Maintains backward compatibility

## Related Files
- Issue documented in: `cost_fix.md`
- Modified files:
  - `jaf/providers/model.py` (~30 lines added/modified)
  - `jaf/core/engine.py` (~15 lines added/modified)

## Date Implemented
November 27, 2025
