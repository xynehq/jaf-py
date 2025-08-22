#!/usr/bin/env python3
"""
Test script for JAF timeout functionality.

This script demonstrates:
1. Tool-specific timeouts
2. RunConfig default timeouts  
3. Global default timeouts
4. Timeout error handling
5. Successful tool execution for comparison
"""

import asyncio
import time
from pydantic import BaseModel
from jaf.core.types import (
    RunConfig, RunState, Agent, Message, ContentRole, 
    generate_run_id, generate_trace_id
)
from jaf.core.tools import create_function_tool, function_tool
from jaf.core.engine import run
from jaf.providers.model import make_litellm_provider


class SlowArgs(BaseModel):
    delay: float
    message: str = "Hello"


class FastArgs(BaseModel):
    message: str


# Tool that intentionally takes time (for timeout testing)
async def slow_operation(args: SlowArgs, context) -> str:
    """A tool that takes a specified amount of time to complete."""
    print(f"[SLOW_TOOL] Starting operation with {args.delay}s delay...")
    await asyncio.sleep(args.delay)
    print(f"[SLOW_TOOL] Completed after {args.delay}s")
    return f"Slow operation completed: {args.message} (took {args.delay}s)"


# Fast tool for comparison
async def fast_operation(args: FastArgs, context) -> str:
    """A tool that completes quickly."""
    print(f"[FAST_TOOL] Quick operation: {args.message}")
    return f"Fast operation: {args.message}"


# Tool with very long timeout (should succeed)
async def patient_operation(args: SlowArgs, context) -> str:
    """A tool with a long timeout that should succeed."""
    print(f"[PATIENT_TOOL] Starting patient operation with {args.delay}s delay...")
    await asyncio.sleep(args.delay)
    print(f"[PATIENT_TOOL] Completed after {args.delay}s")
    return f"Patient operation completed: {args.message} (took {args.delay}s)"


# Create tools with different timeout configurations
def create_test_tools():
    # Tool with 2-second timeout (will timeout)
    quick_timeout_tool = create_function_tool({
        'name': 'quick_timeout_tool',
        'description': 'Tool with 2-second timeout',
        'execute': slow_operation,
        'parameters': SlowArgs,
        'timeout': 2.0  # 2 seconds
    })
    
    # Tool with 10-second timeout (should succeed)
    long_timeout_tool = create_function_tool({
        'name': 'long_timeout_tool', 
        'description': 'Tool with 10-second timeout',
        'execute': patient_operation,
        'parameters': SlowArgs,
        'timeout': 10.0  # 10 seconds
    })
    
    # Tool without timeout (will use RunConfig default)
    default_timeout_tool = create_function_tool({
        'name': 'default_timeout_tool',
        'description': 'Tool using RunConfig default timeout', 
        'execute': slow_operation,
        'parameters': SlowArgs
        # No timeout specified - will use RunConfig default
    })
    
    # Fast tool for successful comparison
    fast_tool = create_function_tool({
        'name': 'fast_tool',
        'description': 'Fast tool that completes quickly',
        'execute': fast_operation,
        'parameters': FastArgs
    })
    
    return [quick_timeout_tool, long_timeout_tool, default_timeout_tool, fast_tool]


# Mock model provider for testing
class MockModelProvider:
    def __init__(self, tool_calls_sequence):
        self.tool_calls_sequence = tool_calls_sequence
        self.call_count = 0
    
    async def get_completion(self, state, agent, config):
        if self.call_count < len(self.tool_calls_sequence):
            tool_calls = self.tool_calls_sequence[self.call_count]
            self.call_count += 1
            return {
                'message': {
                    'content': None,
                    'tool_calls': tool_calls
                }
            }
        else:
            # Final response
            return {
                'message': {
                    'content': 'All timeout tests completed!',
                    'tool_calls': None
                }
            }


async def run_timeout_scenario(scenario_name: str, tools, run_config, tool_calls):
    """Run a specific timeout scenario."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {scenario_name}")
    print(f"{'='*60}")
    
    # Create mock model provider
    model_provider = MockModelProvider(tool_calls)
    
    # Update run config with model provider
    test_config = RunConfig(
        agent_registry=run_config.agent_registry,
        model_provider=model_provider,
        default_tool_timeout=run_config.default_tool_timeout,
        on_event=run_config.on_event
    )
    
    # Create initial state
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content="Test timeout functionality")],
        current_agent_name="test_agent",
        context={},
        turn_count=0
    )
    
    # Run the test
    start_time = time.time()
    result = await run(initial_state, test_config)
    end_time = time.time()
    
    print(f"⏱️  Total execution time: {end_time - start_time:.2f}s")
    print(f"📊 Result status: {result.outcome.status}")
    
    if result.outcome.status == 'completed':
        print(f"✅ Success: {result.outcome.output}")
    else:
        print(f"❌ Error: {result.outcome.error}")
    
    # Print final messages to see tool responses
    print(f"\n📝 Final messages ({len(result.final_state.messages)}):")
    for i, msg in enumerate(result.final_state.messages):
        if msg.role == ContentRole.TOOL:
            print(f"  {i+1}. TOOL: {msg.content[:100]}...")
        elif msg.role == ContentRole.ASSISTANT:
            if msg.tool_calls:
                print(f"  {i+1}. ASSISTANT: Making {len(msg.tool_calls)} tool calls")
            else:
                print(f"  {i+1}. ASSISTANT: {msg.content}")
    
    return result


async def test_all_timeout_scenarios():
    """Run comprehensive timeout tests."""
    print("🚀 JAF Timeout Functionality Test Suite")
    print("=" * 60)
    
    tools = create_test_tools()
    
    # Event handler to track tool executions
    def event_handler(event):
        if event.type == 'tool_call_start':
            print(f"🔧 Starting tool: {event.data.tool_name}")
        elif event.type == 'tool_call_end':
            print(f"✅ Tool {event.data.tool_name} finished.")
    
    # Create test agent
    test_agent = Agent(
        name="test_agent",
        instructions=lambda state: "You are a test agent. Use the provided tools as requested.",
        tools=tools
    )
    
    # Test 1: Tool with 2-second timeout (should timeout)
    print("\n🔥 Test 1: Tool-specific timeout (2s) - SHOULD TIMEOUT")
    run_config_1 = RunConfig(
        agent_registry={"test_agent": test_agent},
        model_provider=None,  # Will be replaced in test
        default_tool_timeout=5.0,  # 5 second default
        on_event=event_handler
    )
    
    tool_calls_1 = [[
        {
            'id': 'call_1',
            'type': 'function',
            'function': {
                'name': 'quick_timeout_tool',
                'arguments': '{"delay": 5.0, "message": "This should timeout"}'
            }
        }
    ]]
    
    await run_timeout_scenario(
        "Tool-specific timeout (2s timeout, 5s operation)",
        tools, run_config_1, tool_calls_1
    )
    
    # Test 2: Tool with long timeout (should succeed)
    print("\n✅ Test 2: Tool-specific long timeout (10s) - SHOULD SUCCEED")
    tool_calls_2 = [[
        {
            'id': 'call_2', 
            'type': 'function',
            'function': {
                'name': 'long_timeout_tool',
                'arguments': '{"delay": 3.0, "message": "This should succeed"}'
            }
        }
    ]]
    
    await run_timeout_scenario(
        "Tool-specific long timeout (10s timeout, 3s operation)",
        tools, run_config_1, tool_calls_2
    )
    
    # Test 3: RunConfig default timeout (should timeout)
    print("\n🔥 Test 3: RunConfig default timeout (3s) - SHOULD TIMEOUT")
    run_config_3 = RunConfig(
        agent_registry={"test_agent": test_agent},
        model_provider=None,
        default_tool_timeout=3.0,  # 3 second default
        on_event=event_handler
    )
    
    tool_calls_3 = [[
        {
            'id': 'call_3',
            'type': 'function', 
            'function': {
                'name': 'default_timeout_tool',
                'arguments': '{"delay": 5.0, "message": "This should timeout with config default"}'
            }
        }
    ]]
    
    await run_timeout_scenario(
        "RunConfig default timeout (3s timeout, 5s operation)",
        tools, run_config_3, tool_calls_3
    )
    
    # Test 4: Global default timeout (30s) 
    print("\n✅ Test 4: Global default timeout (30s) - SHOULD SUCCEED")
    run_config_4 = RunConfig(
        agent_registry={"test_agent": test_agent},
        model_provider=None,
        default_tool_timeout=None,  # No default, will use global 30s
        on_event=event_handler
    )
    
    tool_calls_4 = [[
        {
            'id': 'call_4',
            'type': 'function',
            'function': {
                'name': 'default_timeout_tool', 
                'arguments': '{"delay": 2.0, "message": "This should succeed with global default"}'
            }
        }
    ]]
    
    await run_timeout_scenario(
        "Global default timeout (30s timeout, 2s operation)",
        tools, run_config_4, tool_calls_4
    )
    
    # Test 5: Fast tool (should always succeed)
    print("\n⚡ Test 5: Fast tool - SHOULD SUCCEED")
    tool_calls_5 = [[
        {
            'id': 'call_5',
            'type': 'function',
            'function': {
                'name': 'fast_tool',
                'arguments': '{"message": "Quick operation"}'
            }
        }
    ]]
    
    await run_timeout_scenario(
        "Fast tool (instant completion)",
        tools, run_config_1, tool_calls_5
    )
    
    # Test 6: Multiple tools with mixed timeouts
    print("\n🎯 Test 6: Multiple tools with mixed timeouts")
    tool_calls_6 = [[
        {
            'id': 'call_6a',
            'type': 'function',
            'function': {
                'name': 'fast_tool',
                'arguments': '{"message": "First quick call"}'
            }
        },
        {
            'id': 'call_6b',
            'type': 'function', 
            'function': {
                'name': 'quick_timeout_tool',
                'arguments': '{"delay": 3.0, "message": "This will timeout"}'
            }
        },
        {
            'id': 'call_6c',
            'type': 'function',
            'function': {
                'name': 'fast_tool',
                'arguments': '{"message": "Another quick call"}'
            }
        }
    ]]
    
    await run_timeout_scenario(
        "Multiple tools (1 fast, 1 timeout, 1 fast)",
        tools, run_config_1, tool_calls_6
    )
    
    print(f"\n🎉 TIMEOUT TESTING COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(test_all_timeout_scenarios())
