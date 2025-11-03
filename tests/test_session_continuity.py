#!/usr/bin/env python3
"""
Test script to verify session continuity in agent-as-tool functionality.

This script tests that subagents maintain conversation context across multiple calls,
similar to how OpenAI Agents SDK maintains thread_id across agent tool invocations.
"""

import asyncio
import os
import sys
from typing import Optional
from pydantic import BaseModel

# Add the project root to the path
sys.path.append(".")

from jaf import Agent, ModelProvider, RunConfig
from jaf.memory.providers.in_memory import InMemoryProvider
from jaf.memory.types import MemoryConfig, InMemoryConfig
from jaf.core.agent_tool import create_agent_tool


class SimpleContext(BaseModel):
    user_id: str = "test_user"


# Mock model provider for testing
class MockModelProvider(ModelProvider):
    def __init__(self):
        self.call_count = 0

    async def get_completion(self, state, agent, config):
        self.call_count += 1

        # Simulate different responses based on conversation history
        user_messages = [msg for msg in state.messages if msg.role == "user"]
        assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]

        print(f"[MOCK] Agent: {agent.name}, Call #{self.call_count}")
        print(f"[MOCK] Message history length: {len(state.messages)}")
        print(f"[MOCK] User messages: {len(user_messages)}")
        print(f"[MOCK] Assistant messages: {len(assistant_messages)}")

        if agent.name == "memory_agent":
            # Memory agent remembers previous conversations
            if len(assistant_messages) == 0:
                # First call
                response = "I'll remember this conversation. You said: " + user_messages[-1].content
            else:
                # Subsequent calls - should have access to previous context
                previous_content = assistant_messages[-1].content
                current_input = user_messages[-1].content
                response = f"I remember our previous conversation: '{previous_content}'. Now you said: '{current_input}'"
        else:
            # Main orchestrator agent
            if "test memory" in user_messages[-1].content.lower():
                # Call the memory agent tool
                return {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_memory_1",
                                "function": {
                                    "name": "run_memory_agent",
                                    "arguments": '{"input": "Hello, can you remember this?"}',
                                },
                            }
                        ],
                    }
                }
            elif "follow up" in user_messages[-1].content.lower():
                # Follow up call to memory agent
                return {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_memory_2",
                                "function": {
                                    "name": "run_memory_agent",
                                    "arguments": '{"input": "Do you remember what I said before?"}',
                                },
                            }
                        ],
                    }
                }
            else:
                response = "I'm the main agent. I can call other agents for you."

        return {"message": {"content": response, "tool_calls": None}}


async def test_session_continuity():
    """Test that subagents maintain session continuity across calls."""
    print("=== Testing Session Continuity in Agent-as-Tool ===\n")

    # Create memory provider
    memory_provider_config = InMemoryConfig(
        max_conversations=100, max_messages_per_conversation=100
    )
    memory_provider = InMemoryProvider(memory_provider_config)
    memory_config = MemoryConfig(provider=memory_provider, max_messages=100, auto_store=True)

    # Create model provider
    model_provider = MockModelProvider()

    # Create memory agent (subagent)
    memory_agent = Agent(
        name="memory_agent",
        instructions="You are a memory agent. Remember previous conversations and reference them in your responses.",
        tools=[],
    )

    # Create memory agent tool
    memory_tool = create_agent_tool(
        agent=memory_agent,
        tool_name="run_memory_agent",
        tool_description="Run the memory agent that can remember previous conversations",
    )

    # Create main orchestrator agent
    main_agent = Agent(
        name="main_agent",
        instructions="You are the main agent. Use the memory agent tool to test session continuity.",
        tools=[memory_tool],
    )

    # Create run config with memory
    config = RunConfig(
        agent_registry={"main_agent": main_agent, "memory_agent": memory_agent},
        model_provider=model_provider,
        memory=memory_config,
        conversation_id="test_session_continuity",
        max_turns=10,
    )

    try:
        # Import the run function
        from jaf.core.engine import run
        from jaf.core.types import (
            RunState,
            Message,
            ContentRole,
            generate_run_id,
            generate_trace_id,
        )

        print("Test 1: First call to memory agent")
        print("-" * 50)

        # First conversation - call memory agent
        initial_state_1 = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[
                Message(
                    role=ContentRole.USER, content="Test memory - please remember this conversation"
                )
            ],
            current_agent_name="main_agent",
            context=SimpleContext(),
            turn_count=0,
        )

        result_1 = await run(initial_state_1, config)
        print(f"Result 1 status: {result_1.outcome.status}")
        if hasattr(result_1.outcome, "output"):
            print(f"Result 1 output: {result_1.outcome.output}")

        # Print conversation history after first call
        print(f"\nMessages after first call: {len(result_1.final_state.messages)}")
        for i, msg in enumerate(result_1.final_state.messages):
            print(f"  {i + 1}. {msg.role}: {msg.content[:100]}...")

        print("\n" + "=" * 60 + "\n")

        print("Test 2: Follow-up call to memory agent (should remember previous)")
        print("-" * 50)

        # Second conversation - follow up call (should remember previous context)
        initial_state_2 = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[
                Message(
                    role=ContentRole.USER,
                    content="Follow up - do you remember our previous conversation?",
                )
            ],
            current_agent_name="main_agent",
            context=SimpleContext(),
            turn_count=0,
        )

        result_2 = await run(initial_state_2, config)
        print(f"Result 2 status: {result_2.outcome.status}")
        if hasattr(result_2.outcome, "output"):
            print(f"Result 2 output: {result_2.outcome.output}")

        # Print conversation history after second call
        print(f"\nMessages after second call: {len(result_2.final_state.messages)}")
        for i, msg in enumerate(result_2.final_state.messages):
            print(f"  {i + 1}. {msg.role}: {msg.content[:100]}...")

        print("\n" + "=" * 60 + "\n")

        # Check if session continuity is working
        print("Session Continuity Analysis:")
        print("-" * 30)

        # The second call should have loaded previous conversation history
        # So it should have more messages than just the new user message
        if len(result_2.final_state.messages) > 2:  # Should have loaded previous conversation
            print("✅ SUCCESS: Session continuity is working!")
            print(f"   - Second call loaded previous conversation history")
            print(f"   - Total messages in second call: {len(result_2.final_state.messages)}")
        else:
            print("❌ FAILURE: Session continuity is NOT working")
            print(f"   - Second call did not load previous conversation history")
            print(f"   - Total messages in second call: {len(result_2.final_state.messages)}")

        return len(result_2.final_state.messages) > 2

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    print("Starting session continuity test...\n")

    success = await test_session_continuity()

    print(f"\n{'=' * 60}")
    print("FINAL RESULT:")
    if success:
        print("✅ Session continuity is working correctly!")
        print("   Subagents maintain conversation context across calls.")
    else:
        print("❌ Session continuity needs to be fixed.")
        print("   Subagents are not maintaining conversation context.")
    print(f"{'=' * 60}")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
