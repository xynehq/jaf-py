"""
Example demonstrating LLM callback features for context modification.

This example shows how to use before_llm_call and after_llm_call callbacks
to customize LLM interactions, including context summarization for large contexts.
"""

import asyncio
from dataclasses import dataclass, replace
from typing import List

from jaf.core.types import (
    Agent,
    Message,
    ContentRole,
    RunConfig,
    RunState,
    ModelCompletionResponse,
    generate_run_id,
    generate_trace_id,
)
from jaf.providers.llm_service import LLMService


@dataclass
class SimpleContext:
    """Simple context for demonstration."""
    user_id: str = "user123"
    session_data: dict = None


async def summarize_large_context(
    state: RunState[SimpleContext], 
    agent: Agent
) -> RunState[SimpleContext]:
    """
    Callback that triggers before LLM call.
    Summarizes context if it's too large to save tokens.
    """
    # Calculate approximate token count (rough estimate: 1 token ≈ 4 characters)
    total_chars = sum(len(str(msg.content)) for msg in state.messages)
    estimated_tokens = total_chars // 4
    
    print(f"[BEFORE LLM] Estimated tokens: {estimated_tokens}")
    
    # If context is too large, summarize older messages
    if estimated_tokens > 10000:
        print("[BEFORE LLM] Context too large, summarizing old messages...")
        
        # Keep the system message and recent messages, summarize the middle
        keep_recent = 5  # Keep last 5 messages
        
        if len(state.messages) > keep_recent + 1:
            recent_messages = state.messages[-keep_recent:]
            old_messages = state.messages[:-keep_recent]
            
            # Create a summary of old messages
            summary_text = f"[SUMMARIZED: {len(old_messages)} previous messages about the conversation context]"
            summary_message = Message(
                role=ContentRole.ASSISTANT,
                content=summary_text
            )
            
            # Return state with summarized messages
            new_messages = [summary_message] + recent_messages
            print(f"[BEFORE LLM] Reduced from {len(state.messages)} to {len(new_messages)} messages")
            
            return replace(state, messages=new_messages)
    
    return state


async def log_llm_response(
    state: RunState[SimpleContext],
    response: ModelCompletionResponse
) -> ModelCompletionResponse:
    """
    Callback that triggers after LLM call.
    Logs the response for monitoring/debugging.
    """
    if response.message:
        content_preview = (response.message.content[:100] + "...") if response.message.content and len(response.message.content) > 100 else response.message.content
        print(f"[AFTER LLM] Response preview: {content_preview}")
        
        if response.message.tool_calls:
            print(f"[AFTER LLM] Tool calls: {len(response.message.tool_calls)}")
            for tc in response.message.tool_calls:
                print(f"  - {tc.function.name}")
    
    return response


async def main():
    """Main example function."""
    
    # Create a simple agent
    def instructions(state: RunState) -> str:
        return "You are a helpful assistant. Be concise in your responses."
    
    simple_agent = Agent(
        name="assistant",
        instructions=instructions,
        tools=None,
    )
    
    # Create LLM service (you'll need to set your API key in environment)
    llm_service = LLMService()
    
    # Create RunConfig with callbacks
    config = RunConfig(
        agent_registry={"assistant": simple_agent},
        model_provider=llm_service,
        max_turns=10,
        before_llm_call=summarize_large_context,  # Called before each LLM call
        after_llm_call=log_llm_response,          # Called after each LLM call
    )
    
    # Create initial state with a user message
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[
            Message(
                role=ContentRole.USER,
                content="What is the capital of France?"
            )
        ],
        current_agent_name="assistant",
        context=SimpleContext(user_id="demo_user"),
        turn_count=0,
        approvals={}
    )
    
    # Run the agent
    print("Starting agent execution with callbacks...")
    from jaf.core.engine import run
    result = await run(initial_state, config)
    
    # Print result
    if result.outcome.status == "completed":
        print(f"\n✅ Success! Output: {result.outcome.output}")
    else:
        print(f"\n❌ Error: {result.outcome}")


if __name__ == "__main__":
    asyncio.run(main())
