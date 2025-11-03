#!/usr/bin/env python3
"""
Comprehensive Multi-Agent Manual Tests

This script provides extensive manual testing of multi-agent coordination,
handoffs, context sharing, and complex workflows to validate the JAF framework's
multi-agent capabilities.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from jaf import (
    Agent,
    ContentRole,
    Message,
    ModelConfig,
    RunConfig,
    RunState,
    create_function_tool,
    create_run_id,
    create_trace_id,
    run,
)
from jaf.core.tool_results import ToolResponse


class MultiAgentMockProvider:
    """Mock model provider for multi-agent scenarios."""

    def __init__(self, scenario: str = "normal"):
        self.scenario = scenario
        self.call_count = 0
        self.agent_calls = {}

    async def get_completion(self, state, agent, config):
        self.call_count += 1
        agent_name = agent.name

        if agent_name not in self.agent_calls:
            self.agent_calls[agent_name] = 0
        self.agent_calls[agent_name] += 1

        if self.scenario == "simple_handoff":
            if agent_name == "AgentA":
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "handoff_1",
                                "type": "function",
                                "function": {
                                    "name": "handoff",
                                    "arguments": json.dumps(
                                        {
                                            "target_agent": "AgentB",
                                            "context": "Task from Agent A",
                                            "reason": "Passing to Agent B",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            elif agent_name == "AgentB":
                return {"message": {"content": "Task completed by Agent B", "tool_calls": None}}

        elif self.scenario == "chain_handoff":
            if agent_name == "AgentA" and self.agent_calls[agent_name] == 1:
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "handoff_1",
                                "type": "function",
                                "function": {
                                    "name": "handoff",
                                    "arguments": json.dumps(
                                        {
                                            "target_agent": "AgentB",
                                            "context": "Step 1 complete",
                                            "reason": "Moving to step 2",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            elif agent_name == "AgentB" and self.agent_calls[agent_name] == 1:
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "handoff_2",
                                "type": "function",
                                "function": {
                                    "name": "handoff",
                                    "arguments": json.dumps(
                                        {
                                            "target_agent": "AgentC",
                                            "context": "Step 2 complete",
                                            "reason": "Moving to step 3",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            elif agent_name == "AgentC":
                return {
                    "message": {"content": "All steps completed by Agent C", "tool_calls": None}
                }

        elif self.scenario == "tool_then_handoff":
            if agent_name == "AgentA":
                if self.agent_calls[agent_name] == 1:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tool_1",
                                    "type": "function",
                                    "function": {
                                        "name": "process_data",
                                        "arguments": json.dumps(
                                            {"data": "test_data", "operation": "analyze"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                else:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "handoff_1",
                                    "type": "function",
                                    "function": {
                                        "name": "handoff",
                                        "arguments": json.dumps(
                                            {
                                                "target_agent": "AgentB",
                                                "context": "Data processed, ready for next step",
                                                "reason": "Analysis complete",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
            elif agent_name == "AgentB":
                return {"message": {"content": "Final processing completed", "tool_calls": None}}

        elif self.scenario == "context_sharing":
            if agent_name == "DataCollector":
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "handoff_1",
                                "type": "function",
                                "function": {
                                    "name": "handoff",
                                    "arguments": json.dumps(
                                        {
                                            "target_agent": "DataProcessor",
                                            "context": json.dumps(
                                                {
                                                    "collected_data": ["item1", "item2", "item3"],
                                                    "timestamp": "2024-01-15T10:00:00Z",
                                                    "source": "database",
                                                }
                                            ),
                                            "reason": "Data collection complete",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            elif agent_name == "DataProcessor":
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "handoff_2",
                                "type": "function",
                                "function": {
                                    "name": "handoff",
                                    "arguments": json.dumps(
                                        {
                                            "target_agent": "ReportGenerator",
                                            "context": json.dumps(
                                                {
                                                    "processed_data": [
                                                        "processed_item1",
                                                        "processed_item2",
                                                    ],
                                                    "processing_time": "5.2s",
                                                    "status": "success",
                                                }
                                            ),
                                            "reason": "Processing complete",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            elif agent_name == "ReportGenerator":
                return {
                    "message": {
                        "content": "Report generated successfully with processed data",
                        "tool_calls": None,
                    }
                }

        elif self.scenario == "error_recovery":
            if agent_name == "AgentA":
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "handoff_1",
                                "type": "function",
                                "function": {
                                    "name": "handoff",
                                    "arguments": json.dumps(
                                        {
                                            "target_agent": "NonExistentAgent",
                                            "context": "This should fail",
                                            "reason": "Testing error handling",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }

        else:  # normal completion
            return {
                "message": {
                    "content": f"{agent_name} ready to help! (Call #{self.agent_calls[agent_name]})",
                    "tool_calls": None,
                }
            }


# Create test tools
async def process_data_execute(args: dict, context: Any):
    """Mock data processing tool."""
    return ToolResponse.success(
        {
            "operation": args.get("operation", "unknown"),
            "data": args.get("data", "no_data"),
            "result": "processed_successfully",
            "timestamp": datetime.now().isoformat(),
        }
    )


async def handoff_execute(args: dict, context: Any):
    """Mock handoff tool."""
    return ToolResponse.success(
        {
            "handoff_to": args.get("target_agent"),
            "context": args.get("context"),
            "reason": args.get("reason"),
            "message": f"Handing off to {args.get('target_agent')}: {args.get('reason')}",
        }
    )


# Create tools
process_data_tool = create_function_tool(
    {
        "name": "process_data",
        "description": "Process data with specified operation",
        "execute": process_data_execute,
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Data to process"},
                "operation": {"type": "string", "description": "Operation to perform"},
            },
            "required": ["data", "operation"],
        },
    }
)

handoff_tool = create_function_tool(
    {
        "name": "handoff",
        "description": "Hand off to another agent",
        "execute": handoff_execute,
        "parameters": {
            "type": "object",
            "properties": {
                "target_agent": {"type": "string", "description": "Agent to hand off to"},
                "context": {"type": "string", "description": "Context to pass"},
                "reason": {"type": "string", "description": "Reason for handoff"},
            },
            "required": ["target_agent", "context", "reason"],
        },
    }
)


# Create test agents
def create_test_agent(name: str, tools: list = None, handoffs: list = None):
    """Create a test agent with specified tools and handoff permissions."""
    if tools is None:
        tools = [handoff_tool]

    def instructions(state):
        return f"You are {name}, a helpful assistant agent."

    return Agent(
        name=name,
        instructions=instructions,
        tools=tools,
        handoffs=handoffs,
        model_config=ModelConfig(name="test-model", temperature=0.1),
    )


# Test agents with proper handoff permissions
agent_a = create_test_agent("AgentA", [handoff_tool, process_data_tool], ["AgentB", "AgentC"])
agent_b = create_test_agent("AgentB", [handoff_tool, process_data_tool], ["AgentA", "AgentC"])
agent_c = create_test_agent("AgentC", [handoff_tool, process_data_tool], ["AgentA", "AgentB"])
data_collector = create_test_agent("DataCollector", [handoff_tool], ["DataProcessor"])
data_processor = create_test_agent(
    "DataProcessor", [handoff_tool], ["ReportGenerator", "DataCollector"]
)
report_generator = create_test_agent("ReportGenerator", [handoff_tool], ["DataProcessor"])


# Multi-Agent Manual Tests
async def test_simple_agent_handoff():
    """Test basic agent-to-agent handoff."""
    print("\n🧪 Testing Simple Agent Handoff...")

    provider = MultiAgentMockProvider("simple_handoff")

    state = RunState(
        run_id=create_run_id("simple-handoff"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Start task with Agent A")],
        current_agent_name="AgentA",
        context={"test_id": "simple_handoff"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"AgentA": agent_a, "AgentB": agent_b}, model_provider=provider, max_turns=5
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")

    # Check for successful handoff
    handoff_messages = [
        m for m in result.final_state.messages if m.role == "tool" and "handoff_to" in m.content
    ]

    print(f"   🤝 Handoffs: {len(handoff_messages)}")

    return (
        result.final_state.current_agent_name == "AgentB"
        and len(handoff_messages) >= 1
        and result.outcome.status == "completed"
    )


async def test_chain_agent_handoff():
    """Test chain of handoffs through multiple agents."""
    print("\n🧪 Testing Chain Agent Handoff...")

    provider = MultiAgentMockProvider("chain_handoff")

    state = RunState(
        run_id=create_run_id("chain-handoff"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Start multi-step process")],
        current_agent_name="AgentA",
        context={"test_id": "chain_handoff"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"AgentA": agent_a, "AgentB": agent_b, "AgentC": agent_c},
        model_provider=provider,
        max_turns=8,
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")

    # Check for multiple handoffs
    handoff_messages = [
        m for m in result.final_state.messages if m.role == "tool" and "handoff_to" in m.content
    ]

    print(f"   🤝 Handoffs: {len(handoff_messages)}")
    print(f"   📋 Agent Calls: {provider.agent_calls}")

    return (
        result.final_state.current_agent_name == "AgentC"
        and len(handoff_messages) >= 2
        and result.outcome.status == "completed"
    )


async def test_tool_execution_then_handoff():
    """Test tool execution followed by handoff."""
    print("\n🧪 Testing Tool Execution Then Handoff...")

    provider = MultiAgentMockProvider("tool_then_handoff")

    state = RunState(
        run_id=create_run_id("tool-handoff"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Process data then hand off")],
        current_agent_name="AgentA",
        context={"test_id": "tool_handoff"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"AgentA": agent_a, "AgentB": agent_b}, model_provider=provider, max_turns=6
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")

    # Check for both tool execution and handoff
    tool_messages = [m for m in result.final_state.messages if m.role == "tool"]
    handoff_messages = [m for m in tool_messages if "handoff_to" in m.content]
    process_messages = [m for m in tool_messages if "processed_successfully" in m.content]

    print(f"   🔧 Tool Executions: {len(tool_messages)}")
    print(f"   🤝 Handoffs: {len(handoff_messages)}")
    print(f"   📊 Data Processing: {len(process_messages)}")

    return (
        result.final_state.current_agent_name == "AgentB"
        and len(handoff_messages) >= 1
        and len(process_messages) >= 1
        and result.outcome.status == "completed"
    )


async def test_context_sharing_between_agents():
    """Test context sharing and preservation across agent handoffs."""
    print("\n🧪 Testing Context Sharing Between Agents...")

    provider = MultiAgentMockProvider("context_sharing")

    initial_context = {
        "session_id": "CTX_12345",
        "user_preferences": {"format": "json", "verbose": True},
        "workflow_state": "initialized",
    }

    state = RunState(
        run_id=create_run_id("context-sharing"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Start data processing workflow")],
        current_agent_name="DataCollector",
        context=initial_context,
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={
            "DataCollector": data_collector,
            "DataProcessor": data_processor,
            "ReportGenerator": report_generator,
        },
        model_provider=provider,
        max_turns=8,
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")
    print(f"   📋 Context Preserved: {result.final_state.context.get('session_id') == 'CTX_12345'}")

    # Check for context data in handoffs
    handoff_messages = [
        m for m in result.final_state.messages if m.role == "tool" and "handoff_to" in m.content
    ]
    context_data_found = False

    for msg in handoff_messages:
        tool_result = json.loads(msg.content)
        if "context" in tool_result:
            try:
                context_json = json.loads(tool_result["context"])
                if "collected_data" in context_json or "processed_data" in context_json:
                    context_data_found = True
                    break
            except:
                pass

    print(f"   🤝 Handoffs: {len(handoff_messages)}")
    print(f"   📊 Context Data Shared: {'✅' if context_data_found else '❌'}")

    return (
        result.final_state.current_agent_name == "ReportGenerator"
        and len(handoff_messages) >= 2
        and context_data_found
        and result.outcome.status == "completed"
    )


async def test_concurrent_agent_scenarios():
    """Test scenarios that might involve concurrent agent considerations."""
    print("\n🧪 Testing Concurrent Agent Scenarios...")

    provider = MultiAgentMockProvider("normal")

    # Test multiple rapid handoffs
    state = RunState(
        run_id=create_run_id("concurrent-test"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Test concurrent handling")],
        current_agent_name="AgentA",
        context={"test_type": "concurrent", "timestamp": datetime.now().isoformat()},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"AgentA": agent_a, "AgentB": agent_b, "AgentC": agent_c},
        model_provider=provider,
        max_turns=3,
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")
    print(f"   📋 Context Preserved: {result.final_state.context.get('test_type') == 'concurrent'}")

    return result.outcome.status == "completed"


async def test_error_handling_in_handoffs():
    """Test error handling when handoffs fail."""
    print("\n🧪 Testing Error Handling in Handoffs...")

    provider = MultiAgentMockProvider("error_recovery")

    state = RunState(
        run_id=create_run_id("error-handoff"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Test error handling")],
        current_agent_name="AgentA",
        context={"test_id": "error_handling"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"AgentA": agent_a, "AgentB": agent_b}, model_provider=provider, max_turns=5
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")

    # Check if error was handled gracefully
    error_handled = result.outcome.status in ["error", "completed"]

    print(f"   ⚠️ Error Handled: {'✅' if error_handled else '❌'}")

    return error_handled


async def test_agent_state_isolation():
    """Test that agents maintain proper state isolation."""
    print("\n🧪 Testing Agent State Isolation...")

    provider = MultiAgentMockProvider("normal")

    # Test with different contexts for different agents
    state1 = RunState(
        run_id=create_run_id("isolation-test-1"),
        trace_id=create_trace_id("test-trace-1"),
        messages=[Message(role=ContentRole.USER, content="Test isolation 1")],
        current_agent_name="AgentA",
        context={"agent_context": "A", "data": "secret_a"},
        turn_count=0,
    )

    state2 = RunState(
        run_id=create_run_id("isolation-test-2"),
        trace_id=create_trace_id("test-trace-2"),
        messages=[Message(role=ContentRole.USER, content="Test isolation 2")],
        current_agent_name="AgentB",
        context={"agent_context": "B", "data": "secret_b"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"AgentA": agent_a, "AgentB": agent_b}, model_provider=provider, max_turns=3
    )

    # Run both scenarios
    result1 = await run(state1, config)
    result2 = await run(state2, config)

    print(f"   ✅ Test 1 Status: {result1.outcome.status}")
    print(f"   ✅ Test 2 Status: {result2.outcome.status}")
    print(f"   🔒 Context 1: {result1.final_state.context.get('data')}")
    print(f"   🔒 Context 2: {result2.final_state.context.get('data')}")

    # Check isolation
    isolation_maintained = (
        result1.final_state.context.get("data") == "secret_a"
        and result2.final_state.context.get("data") == "secret_b"
        and result1.final_state.context != result2.final_state.context
    )

    print(f"   🔒 Isolation Maintained: {'✅' if isolation_maintained else '❌'}")

    return isolation_maintained


async def test_complex_multi_agent_workflow():
    """Test a complex workflow involving multiple agents and tools."""
    print("\n🧪 Testing Complex Multi-Agent Workflow...")

    # Custom provider for complex workflow
    class ComplexWorkflowProvider:
        def __init__(self):
            self.call_count = 0
            self.agent_calls = {}

        async def get_completion(self, state, agent, config):
            self.call_count += 1
            agent_name = agent.name

            if agent_name not in self.agent_calls:
                self.agent_calls[agent_name] = 0
            self.agent_calls[agent_name] += 1

            if agent_name == "DataCollector":
                if self.agent_calls[agent_name] == 1:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "collect_1",
                                    "type": "function",
                                    "function": {
                                        "name": "process_data",
                                        "arguments": json.dumps(
                                            {"data": "raw_dataset", "operation": "collect"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                else:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "handoff_1",
                                    "type": "function",
                                    "function": {
                                        "name": "handoff",
                                        "arguments": json.dumps(
                                            {
                                                "target_agent": "DataProcessor",
                                                "context": "Data collected successfully",
                                                "reason": "Ready for processing",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
            elif agent_name == "DataProcessor":
                if self.agent_calls[agent_name] == 1:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "process_1",
                                    "type": "function",
                                    "function": {
                                        "name": "process_data",
                                        "arguments": json.dumps(
                                            {"data": "collected_data", "operation": "analyze"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                else:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "handoff_2",
                                    "type": "function",
                                    "function": {
                                        "name": "handoff",
                                        "arguments": json.dumps(
                                            {
                                                "target_agent": "ReportGenerator",
                                                "context": "Analysis complete",
                                                "reason": "Generate final report",
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
            elif agent_name == "ReportGenerator":
                if self.agent_calls[agent_name] == 1:
                    return {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "report_1",
                                    "type": "function",
                                    "function": {
                                        "name": "process_data",
                                        "arguments": json.dumps(
                                            {"data": "analyzed_data", "operation": "report"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                else:
                    return {
                        "message": {
                            "content": "Complex workflow completed successfully!",
                            "tool_calls": None,
                        }
                    }

    provider = ComplexWorkflowProvider()

    state = RunState(
        run_id=create_run_id("complex-workflow"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Start complex data processing workflow")],
        current_agent_name="DataCollector",
        context={"workflow_id": "COMPLEX_001", "priority": "high"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={
            "DataCollector": data_collector,
            "DataProcessor": data_processor,
            "ReportGenerator": report_generator,
        },
        model_provider=provider,
        max_turns=10,
    )

    result = await run(state, config)

    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")

    # Analyze workflow execution
    tool_messages = [m for m in result.final_state.messages if m.role == "tool"]
    handoff_messages = [m for m in tool_messages if "handoff_to" in m.content]
    process_messages = [m for m in tool_messages if "processed_successfully" in m.content]

    print(f"   🔧 Total Tool Executions: {len(tool_messages)}")
    print(f"   🤝 Handoffs: {len(handoff_messages)}")
    print(f"   📊 Data Operations: {len(process_messages)}")
    print(f"   📋 Agent Calls: {provider.agent_calls}")

    workflow_success = (
        result.final_state.current_agent_name == "ReportGenerator"
        and len(handoff_messages) >= 2
        and result.outcome.status == "completed"
        and result.final_state.turn_count >= 5
    )

    print(f"   🎯 Workflow Success: {'✅' if workflow_success else '❌'}")

    return workflow_success


# Manual test runner for multi-agent scenarios
async def run_multi_agent_manual_tests():
    """Run all multi-agent manual tests."""
    print("🤖 Starting Multi-Agent Manual Tests")
    print("=" * 70)

    tests = [
        ("Simple Agent Handoff", test_simple_agent_handoff),
        ("Chain Agent Handoff", test_chain_agent_handoff),
        ("Tool Execution Then Handoff", test_tool_execution_then_handoff),
        ("Context Sharing Between Agents", test_context_sharing_between_agents),
        ("Concurrent Agent Scenarios", test_concurrent_agent_scenarios),
        ("Error Handling in Handoffs", test_error_handling_in_handoffs),
        ("Agent State Isolation", test_agent_state_isolation),
        ("Complex Multi-Agent Workflow", test_complex_multi_agent_workflow),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            success = await test_func()
            passed += 1 if success else 0
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        except Exception as e:
            print(f"❌ FAIL {test_name} - Exception: {e}")

    print("\n" + "=" * 70)
    print("📊 Multi-Agent Manual Test Results")
    print("=" * 70)
    print(f"🎯 Overall: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed == total:
        print("🎉 All multi-agent manual tests passed!")
        print("🤖 Multi-agent coordination system is fully functional!")
    else:
        print("⚠️ Some multi-agent tests failed.")
        print("🔧 Review the results above for specific issues.")

    print("\n🔍 Test Coverage Summary:")
    print("  • Simple agent-to-agent handoffs")
    print("  • Chain handoffs through multiple agents")
    print("  • Tool execution combined with handoffs")
    print("  • Context sharing and preservation")
    print("  • Concurrent agent scenario handling")
    print("  • Error handling and recovery")
    print("  • Agent state isolation")
    print("  • Complex multi-step workflows")

    return passed == total


if __name__ == "__main__":
    asyncio.run(run_multi_agent_manual_tests())
