#!/usr/bin/env python3
"""
Manual Core Engine Tests for JAF Python Framework

This module provides comprehensive manual testing of the core JAF engine
functionality, focusing on the core engine module specifically.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, Field

from jaf.core.engine import run
from jaf.core.types import (
    Agent,
    ContentRole,
    Message,
    ModelConfig,
    RunConfig,
    RunState,
    ToolSource,
    create_run_id,
    create_trace_id,
)
from jaf.core.tools import create_function_tool
from jaf.core.tool_results import ToolResponse, ToolResult


# Test Tool Definitions for Core Engine Testing
class CalculatorArgs(BaseModel):
    """Calculator tool arguments."""
    operation: str = Field(description="Operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class HandoffArgs(BaseModel):
    """Handoff tool arguments."""
    target_agent: str = Field(description="Target agent name")
    reason: str = Field(description="Reason for handoff")


# Core Tool Implementations
async def calculator_execute(args: CalculatorArgs, context: Any) -> ToolResult:
    """Execute calculator operations."""
    try:
        if args.operation == "add":
            result = args.a + args.b
        elif args.operation == "subtract":
            result = args.a - args.b
        elif args.operation == "multiply":
            result = args.a * args.b
        elif args.operation == "divide":
            if args.b == 0:
                return ToolResponse.validation_error("Cannot divide by zero")
            result = args.a / args.b
        else:
            return ToolResponse.validation_error(f"Unknown operation: {args.operation}")
        
        return ToolResponse.success({
            "operation": args.operation,
            "operands": [args.a, args.b],
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return ToolResponse.error(f"Calculator error: {str(e)}")


async def handoff_execute(args: HandoffArgs, context: Any) -> ToolResult:
    """Execute agent handoff."""
    return ToolResponse.success({
        "handoff_to": args.target_agent,
        "reason": args.reason,
        "timestamp": datetime.now().isoformat(),
        "context_preserved": True
    })


# Create core tools
calculator_tool = create_function_tool({
    'name': 'calculator',
    'description': 'Perform mathematical calculations',
    'execute': calculator_execute,
    'parameters': CalculatorArgs,
    'source': ToolSource.NATIVE
})

handoff_tool = create_function_tool({
    'name': 'handoff_to_agent',
    'description': 'Hand off to another agent',
    'execute': handoff_execute,
    'parameters': HandoffArgs,
    'source': ToolSource.NATIVE
})


# Mock Model Provider for Core Engine Testing
class CoreMockModelProvider:
    """Mock model provider specifically for core engine testing."""
    
    def __init__(self, scenario: str = "normal"):
        self.scenario = scenario
        self.call_count = 0
    
    async def get_completion(self, state, agent, config):
        self.call_count += 1
        
        if self.scenario == "tool_call":
            if self.call_count == 1:
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'calc_1',
                                'type': 'function',
                                'function': {
                                    'name': 'calculator',
                                    'arguments': json.dumps({
                                        'operation': 'add',
                                        'a': 10,
                                        'b': 5
                                    })
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    'message': {
                        'content': 'Tool execution completed successfully.',
                        'tool_calls': None
                    }
                }
        elif self.scenario == "handoff":
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'handoff_1',
                            'type': 'function',
                            'function': {
                                'name': 'handoff_to_agent',
                                'arguments': json.dumps({
                                    'target_agent': 'TargetAgent',
                                    'reason': 'Testing handoff functionality'
                                })
                            }
                        }
                    ]
                }
            }
        elif self.scenario == "error_tool":
            if self.call_count == 1:
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'calc_error',
                                'type': 'function',
                                'function': {
                                    'name': 'calculator',
                                    'arguments': json.dumps({
                                        'operation': 'divide',
                                        'a': 10,
                                        'b': 0
                                    })
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    'message': {
                        'content': 'Error was handled properly.',
                        'tool_calls': None
                    }
                }
        else:  # normal text completion
            return {
                'message': {
                    'content': f'Core engine response #{self.call_count} from {agent.name}.',
                    'tool_calls': None
                }
            }


def create_core_test_agent(name: str, tools: List = None, handoffs: List[str] = None) -> Agent:
    """Create a test agent for core engine testing."""
    
    def agent_instructions(state) -> str:
        return f"You are {name}, a test agent for core engine validation."
    
    return Agent(
        name=name,
        instructions=agent_instructions,
        tools=tools or [],
        handoffs=handoffs,
        model_config=ModelConfig(name="test-model", temperature=0.0)
    )


# Core Engine Manual Tests
@pytest.mark.asyncio
async def test_core_engine_basic_completion():
    """Test core engine basic text completion."""
    print("\n🧪 Testing Core Engine Basic Completion...")
    
    agent = create_core_test_agent("CoreTestAgent")
    provider = CoreMockModelProvider("normal")
    
    state = RunState(
        run_id=create_run_id("test-basic"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Hello core engine")],
        current_agent_name="CoreTestAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"CoreTestAgent": agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   📝 Output: {result.outcome.output}")
    print(f"   🔄 Turns: {result.final_state.turn_count}")
    
    assert result.outcome.status == "completed"
    assert "Core engine response" in result.outcome.output
    assert result.final_state.turn_count == 1


@pytest.mark.asyncio
async def test_core_engine_tool_execution():
    """Test core engine tool execution."""
    print("\n🧪 Testing Core Engine Tool Execution...")
    
    agent = create_core_test_agent("ToolTestAgent", [calculator_tool])
    provider = CoreMockModelProvider("tool_call")
    
    state = RunState(
        run_id=create_run_id("test-tool"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Calculate 10 + 5")],
        current_agent_name="ToolTestAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"ToolTestAgent": agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   🔄 Turns: {result.final_state.turn_count}")
    
    # Check for tool execution in messages
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    assert len(tool_messages) == 1
    
    tool_result = json.loads(tool_messages[0].content)
    print(f"   🔧 Tool Result: {tool_result.get('result', 'N/A')}")
    
    assert tool_result.get('result') == 15.0  # 10 + 5 = 15
    assert tool_result.get('operation') == 'add'


@pytest.mark.asyncio
async def test_core_engine_agent_handoff():
    """Test core engine agent handoff functionality."""
    print("\n🧪 Testing Core Engine Agent Handoff...")
    
    source_agent = create_core_test_agent("SourceAgent", [handoff_tool], ["TargetAgent"])
    target_agent = create_core_test_agent("TargetAgent")
    
    provider = CoreMockModelProvider("handoff")
    
    state = RunState(
        run_id=create_run_id("test-handoff"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Please handoff to target agent")],
        current_agent_name="SourceAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={
            "SourceAgent": source_agent,
            "TargetAgent": target_agent
        },
        model_provider=provider,
        max_turns=5
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Turns: {result.final_state.turn_count}")
    
    assert result.final_state.current_agent_name == "TargetAgent"
    assert result.final_state.turn_count >= 1


@pytest.mark.asyncio
async def test_core_engine_error_handling():
    """Test core engine error handling."""
    print("\n🧪 Testing Core Engine Error Handling...")
    
    agent = create_core_test_agent("ErrorTestAgent", [calculator_tool])
    provider = CoreMockModelProvider("error_tool")
    
    state = RunState(
        run_id=create_run_id("test-error"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Divide by zero")],
        current_agent_name="ErrorTestAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"ErrorTestAgent": agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    
    # Check for error handling in tool messages
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    assert len(tool_messages) == 1
    
    tool_content = tool_messages[0].content
    print(f"   ⚠️ Error Handled: {'Cannot divide by zero' in tool_content}")
    
    assert "Cannot divide by zero" in tool_content
    assert "validation_error" in tool_content


@pytest.mark.asyncio
async def test_core_engine_context_preservation():
    """Test core engine context preservation."""
    print("\n🧪 Testing Core Engine Context Preservation...")
    
    agent = create_core_test_agent("ContextTestAgent")
    provider = CoreMockModelProvider("normal")
    
    initial_context = {
        "user_id": "core_test_user",
        "session_id": "core_test_session",
        "test_data": {"key": "value"}
    }
    
    state = RunState(
        run_id=create_run_id("test-context"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Test context preservation")],
        current_agent_name="ContextTestAgent",
        context=initial_context,
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"ContextTestAgent": agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   📋 Context Preserved: {result.final_state.context == initial_context}")
    
    assert result.final_state.context == initial_context
    assert result.outcome.status == "completed"


@pytest.mark.asyncio
async def test_core_engine_max_turns():
    """Test core engine max turns limit."""
    print("\n🧪 Testing Core Engine Max Turns Limit...")
    
    # Create a provider that always calls tools to exceed max turns
    class InfiniteToolProvider:
        def __init__(self):
            self.call_count = 0
        
        async def get_completion(self, state, agent, config):
            self.call_count += 1
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'id': f'calc_{self.call_count}',
                            'type': 'function',
                            'function': {
                                'name': 'calculator',
                                'arguments': json.dumps({
                                    'operation': 'add',
                                    'a': self.call_count,
                                    'b': 1
                                })
                            }
                        }
                    ]
                }
            }
    
    agent = create_core_test_agent("MaxTurnsAgent", [calculator_tool])
    provider = InfiniteToolProvider()
    
    state = RunState(
        run_id=create_run_id("test-max-turns"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Keep calculating")],
        current_agent_name="MaxTurnsAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"MaxTurnsAgent": agent},
        model_provider=provider,
        max_turns=2  # Low limit to test
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   🔄 Final Turns: {result.final_state.turn_count}")
    
    assert result.outcome.status == "error"
    assert result.final_state.turn_count >= 2


# Manual test runner for core engine
async def run_core_engine_manual_tests():
    """Run all core engine manual tests."""
    print("🚀 Starting Core Engine Manual Tests")
    print("=" * 60)
    
    tests = [
        test_core_engine_basic_completion,
        test_core_engine_tool_execution,
        test_core_engine_agent_handoff,
        test_core_engine_error_handling,
        test_core_engine_context_preservation,
        test_core_engine_max_turns,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            await test_func()
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Core Engine Manual Test Results")
    print("=" * 60)
    print(f"🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All core engine manual tests passed!")
    else:
        print("⚠️ Some core engine tests failed.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_core_engine_manual_tests())
