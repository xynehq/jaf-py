#!/usr/bin/env python3
"""
Focused demonstration of tool execution callbacks and hooks.

This script specifically demonstrates tool-related callbacks working
with proper function call parsing and execution.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import JAF core types
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaf.core.types import Message, Agent, Tool, ToolSchema, ToolCall, ToolCallFunction

# Import ADK types
from adk.runners.types import RunnerConfig, AgentResponse, RunContext
from adk.runners.agent_runner import execute_agent


# ============================================================================
# SIMPLE TOOL IMPLEMENTATION
# ============================================================================


@dataclass
class CalculatorArgs:
    operation: str
    a: float
    b: float


class CalculatorTool:
    """Simple calculator tool for demonstration."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="Perform basic mathematical operations",
            parameters=CalculatorArgs,
        )

    async def execute(self, args: Any, context: Any) -> str:
        # Handle both dict and dataclass args
        if isinstance(args, dict):
            operation = args.get("operation", "add")
            a = args.get("a", 0)
            b = args.get("b", 0)
        else:
            operation = args.operation
            a = args.a
            b = args.b

        await asyncio.sleep(0.1)  # Simulate processing

        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero",
        }

        result = operations.get(operation, "Error: Unknown operation")

        return json.dumps(
            {"operation": operation, "operands": [a, b], "result": result, "success": True}
        )


# ============================================================================
# CALLBACK IMPLEMENTATION WITH DETAILED LOGGING
# ============================================================================


class ToolCallbackDemo:
    """Callback implementation focused on tool execution demonstration."""

    def __init__(self):
        self.events = []
        self.start_time = time.time()

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log an event with timestamp and details."""
        timestamp = time.time() - self.start_time
        event = {"timestamp": f"{timestamp:.3f}s", "type": event_type, "details": details}
        self.events.append(event)

        print(f"🔔 [{timestamp:.3f}s] {event_type}")
        for key, value in details.items():
            print(f"   📋 {key}: {value}")

    # ========== Lifecycle Hooks ==========

    async def on_start(self, context: RunContext, message: Message, session_state: Dict[str, Any]):
        self.log_event(
            "LIFECYCLE_START",
            {
                "user_query": message.content,
                "context_keys": list(context.keys()) if context else [],
                "session_state_size": len(session_state),
            },
        )

    async def on_complete(self, response: AgentResponse):
        self.log_event(
            "LIFECYCLE_COMPLETE",
            {
                "final_response": response.content.content,
                "execution_time_ms": response.execution_time_ms,
                "tool_calls_made": response.metadata.get("tool_calls_count", 0),
            },
        )

    async def on_error(self, error: Exception, context: RunContext):
        self.log_event(
            "LIFECYCLE_ERROR", {"error_type": type(error).__name__, "error_message": str(error)}
        )

    # ========== LLM Interaction Hooks ==========

    async def on_before_llm_call(
        self, agent: Agent, message: Message, session_state: Dict[str, Any]
    ):
        self.log_event(
            "LLM_BEFORE_CALL",
            {
                "agent_name": agent.name,
                "message_role": message.role,
                "message_content": message.content[:100] + "..."
                if len(message.content) > 100
                else message.content,
                "available_tools": [tool.schema.name for tool in (agent.tools or [])],
            },
        )
        return None

    async def on_after_llm_call(self, response: Message, session_state: Dict[str, Any]):
        tool_calls_info = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_calls_info.append(
                    {"id": tc.id, "tool_name": tc.function.name, "arguments": tc.function.arguments}
                )

        self.log_event(
            "LLM_AFTER_CALL",
            {
                "response_role": response.role,
                "response_content": response.content[:100] + "..."
                if response.content and len(response.content) > 100
                else response.content,
                "has_tool_calls": bool(response.tool_calls),
                "tool_calls": tool_calls_info,
            },
        )
        return response

    # ========== Tool Execution Hooks ==========

    async def on_before_tool_selection(self, available_tools: List[Tool], context_data: List[Any]):
        self.log_event(
            "TOOL_BEFORE_SELECTION",
            {
                "available_tools": [tool.schema.name for tool in available_tools],
                "context_data_count": len(context_data),
            },
        )
        return None

    async def on_tool_selected(self, tool_name: str, args: Dict[str, Any]):
        self.log_event("TOOL_SELECTED", {"tool_name": tool_name, "arguments": args})

    async def on_before_tool_execution(self, tool: Tool, params: Dict[str, Any]):
        self.log_event(
            "TOOL_BEFORE_EXECUTION",
            {
                "tool_name": tool.schema.name,
                "tool_description": tool.schema.description,
                "parameters": params,
            },
        )
        return None

    async def on_after_tool_execution(self, tool: Tool, result: Dict[str, Any]):
        self.log_event(
            "TOOL_AFTER_EXECUTION",
            {
                "tool_name": tool.schema.name,
                "execution_success": result.get("success", False),
                "result_data": result.get("data"),
                "error": result.get("error"),
            },
        )
        return result

    # ========== Iteration Control ==========

    async def on_iteration_start(self, iteration_count: int):
        self.log_event("ITERATION_START", {"iteration_number": iteration_count})
        return None

    async def on_iteration_complete(self, iteration_count: int, had_tool_calls: bool):
        self.log_event(
            "ITERATION_COMPLETE",
            {"iteration_number": iteration_count, "had_tool_calls": had_tool_calls},
        )
        return None


# ============================================================================
# MOCK MODEL PROVIDER WITH TOOL CALLS
# ============================================================================


class ToolCallModelProvider:
    """Mock model provider that generates proper tool calls."""

    def __init__(self):
        self.call_count = 0

    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(0.1)

        # Get the last user message
        user_messages = [msg for msg in state.messages if msg.role == "user"]
        if not user_messages:
            return {"message": {"content": "I need a question to help you with."}}

        last_message = user_messages[-1].content.lower()

        # Generate tool calls based on the query
        if (
            "calculate" in last_message
            or "math" in last_message
            or any(
                op in last_message
                for op in ["add", "multiply", "divide", "subtract", "+", "*", "/", "-"]
            )
        ):
            # Extract numbers and operation from the message
            if "15" in last_message and "25" in last_message:
                operation = "add" if "+" in last_message or "add" in last_message else "multiply"
                a, b = 15, 25
            elif "7" in last_message and "8" in last_message:
                operation = (
                    "multiply" if "*" in last_message or "multiply" in last_message else "add"
                )
                a, b = 7, 8
            else:
                operation = "add"
                a, b = 10, 5

            return {
                "message": {
                    "content": f"I'll calculate {a} {operation} {b} for you.",
                    "tool_calls": [
                        {
                            "id": f"call_calc_{self.call_count}",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"operation": operation, "a": a, "b": b}),
                            },
                        }
                    ],
                }
            }

        # Default response without tool calls
        return {"message": {"content": "I understand your request. Let me help you with that."}}


# ============================================================================
# DEMO EXECUTION
# ============================================================================


async def run_tool_callback_demo():
    """Run focused demonstration of tool callbacks."""

    print("🚀 Tool Callback and Hooks Demonstration")
    print("=" * 60)

    # Create calculator tool
    calculator = CalculatorTool()

    # Create agent with the tool
    def agent_instructions(state):
        return "You are a helpful calculator assistant. Use the calculator tool to perform mathematical operations."

    agent = Agent(name="calculator_agent", instructions=agent_instructions, tools=[calculator])

    # Create callback handler
    callbacks = ToolCallbackDemo()

    # Create session provider
    class MockSessionProvider:
        async def get_session(self, session_id: str):
            return {}

        async def save_session(self, session_id: str, data: dict):
            pass

    # Create configuration
    config = RunnerConfig(
        agent=agent,
        session_provider=MockSessionProvider(),
        callbacks=callbacks,
        max_llm_calls=3,
        enable_context_accumulation=True,
        enable_loop_detection=True,
    )

    # Create model provider
    model_provider = ToolCallModelProvider()

    # Test queries that will trigger tool calls
    test_queries = ["Please calculate 15 + 25", "What is 7 * 8?", "Can you help me with some math?"]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🧮 TEST {i}: {query}")
        print("-" * 40)

        # Reset callback events
        callbacks.events = []
        callbacks.start_time = time.time()

        # Create message and context
        message = Message(role="user", content=query)
        context = {"user_id": f"test_user_{i}", "session_id": f"test_session_{i}"}

        try:
            # Execute the agent
            response = await execute_agent(
                config=config,
                session_state={},
                message=message,
                context=context,
                model_provider=model_provider,
            )

            print(f"\n✅ EXECUTION COMPLETED")
            print(f"📤 Response: {response.content.content}")
            print(f"⏱️  Time: {response.execution_time_ms:.2f}ms")

        except Exception as e:
            print(f"\n❌ EXECUTION FAILED: {e}")
            import traceback

            traceback.print_exc()

        # Show event summary
        print(f"\n📊 CALLBACK EVENTS ({len(callbacks.events)} total):")
        for event in callbacks.events:
            print(f"   {event['timestamp']} - {event['type']}")

        print("\n" + "=" * 60)
        await asyncio.sleep(0.3)

    print("\n🎉 Tool callback demonstration completed!")
    print("\n📈 DEMONSTRATED FEATURES:")
    print("   ✅ Function call parsing from LLM responses")
    print("   ✅ Tool selection and execution hooks")
    print("   ✅ Parameter passing and result handling")
    print("   ✅ Lifecycle management (start, complete, error)")
    print("   ✅ LLM interaction hooks (before/after calls)")
    print("   ✅ Iteration control hooks")
    print("   ✅ Comprehensive event logging and monitoring")


if __name__ == "__main__":
    asyncio.run(run_tool_callback_demo())
