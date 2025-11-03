#!/usr/bin/env python3
"""
Test suite for code examples in docs/api-reference.md
This ensures all API reference examples work with the actual implementation.
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

# Test imports from the API reference examples
try:
    import jaf
    from jaf import RunState, Agent, Message, RunConfig
    from jaf import ContentRole, ToolSource, Model, ToolParameterType, PartType
    from jaf import create_function_tool, ToolResponse, ToolResult, tool_result_to_string
    from jaf import generate_run_id, generate_trace_id, create_run_id, create_trace_id
    from jaf.core.types import FunctionToolConfig, ToolExecuteFunction
    from pydantic import BaseModel, Field, field_validator

    print("✅ All API reference imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class ResultTracker:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def success(self, test_name):
        self.passed += 1
        print(f"✅ {test_name}")

    def failure(self, test_name, error):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n📊 Test Results: {self.passed}/{total} passed")
        if self.errors:
            print("\n❌ Failures:")
            for error in self.errors:
                print(f"   - {error}")


results = ResultTracker()


def test_enums():
    """Test enum definitions from API reference."""

    # Test ContentRole enum
    try:
        assert ContentRole.USER == "user"
        assert ContentRole.ASSISTANT == "assistant"
        assert ContentRole.TOOL == "tool"
        assert ContentRole.SYSTEM == "system"

        message = Message(role=ContentRole.USER, content="Hello!")
        assert message.role == "user"
        results.success("api-reference: ContentRole enum")
    except Exception as e:
        results.failure("api-reference: ContentRole enum", str(e))

    # Test ToolSource enum
    try:
        assert ToolSource.NATIVE == "native"
        assert ToolSource.MCP == "mcp"
        assert ToolSource.PLUGIN == "plugin"
        assert ToolSource.EXTERNAL == "external"
        results.success("api-reference: ToolSource enum")
    except Exception as e:
        results.failure("api-reference: ToolSource enum", str(e))

    # Test Model enum
    try:
        assert Model.GPT_4 == "gpt-4"
        assert Model.GEMINI_PRO == "gemini-pro"
        assert Model.CLAUDE_3_SONNET == "claude-3-sonnet"
        results.success("api-reference: Model enum")
    except Exception as e:
        results.failure("api-reference: Model enum", str(e))

    # Test ToolParameterType enum
    try:
        assert ToolParameterType.STRING == "string"
        assert ToolParameterType.NUMBER == "number"
        assert ToolParameterType.BOOLEAN == "boolean"
        results.success("api-reference: ToolParameterType enum")
    except Exception as e:
        results.failure("api-reference: ToolParameterType enum", str(e))

    # Test PartType enum
    try:
        assert PartType.TEXT == "text"
        assert PartType.IMAGE == "image"
        assert PartType.AUDIO == "audio"
        results.success("api-reference: PartType enum")
    except Exception as e:
        results.failure("api-reference: PartType enum", str(e))


def test_tool_creation():
    """Test tool creation examples from API reference."""

    # Test create_function_tool with object config
    try:

        class GreetArgs(BaseModel):
            name: str = Field(description="Name to greet")

        async def greet_execute(args: GreetArgs, context) -> str:
            return f"Hello, {args.name}!"

        tool = create_function_tool(
            {
                "name": "greet",
                "description": "Greets a user by name",
                "execute": greet_execute,
                "parameters": GreetArgs,
                "metadata": {"category": "social"},
                "source": ToolSource.NATIVE,
            }
        )

        assert tool.schema.name == "greet"
        assert tool.schema.description == "Greets a user by name"
        results.success("api-reference: create_function_tool object config")
    except Exception as e:
        results.failure("api-reference: create_function_tool object config", str(e))


def test_id_generation():
    """Test ID generation functions from API reference."""

    # Test generate_run_id
    try:
        run_id = generate_run_id()
        assert isinstance(run_id, str)
        # The actual implementation uses UUID, not 'run_' prefix
        assert len(str(run_id)) > 0
        results.success("api-reference: generate_run_id")
    except Exception as e:
        results.failure("api-reference: generate_run_id", str(e))

    # Test generate_trace_id
    try:
        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)
        # The actual implementation uses UUID, not 'trace_' prefix
        assert len(str(trace_id)) > 0
        results.success("api-reference: generate_trace_id")
    except Exception as e:
        results.failure("api-reference: generate_trace_id", str(e))

    # Test create_run_id
    try:
        run_id = create_run_id("test_run_123")
        assert str(run_id) == "test_run_123"
        results.success("api-reference: create_run_id")
    except Exception as e:
        results.failure("api-reference: create_run_id", str(e))

    # Test create_trace_id
    try:
        trace_id = create_trace_id("test_trace_456")
        assert str(trace_id) == "test_trace_456"
        results.success("api-reference: create_trace_id")
    except Exception as e:
        results.failure("api-reference: create_trace_id", str(e))


def test_core_types():
    """Test core type definitions from API reference."""

    # Test RunState creation
    try:

        @dataclass
        class MyContext:
            user_id: str
            permissions: List[str]

        state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role="user", content="Hello!")],
            current_agent_name="assistant",
            context=MyContext(user_id="123", permissions=["read"]),
            turn_count=0,
        )

        assert state.current_agent_name == "assistant"
        assert state.context.user_id == "123"
        assert len(state.messages) == 1
        results.success("api-reference: RunState creation")
    except Exception as e:
        results.failure("api-reference: RunState creation", str(e))

    # Test Message creation
    try:
        user_message = Message(role="user", content="What is 2+2?")
        assistant_message = Message(role="assistant", content="Let me calculate that for you.")

        assert user_message.role == "user"
        assert assistant_message.content == "Let me calculate that for you."
        results.success("api-reference: Message creation")
    except Exception as e:
        results.failure("api-reference: Message creation", str(e))


def test_tool_protocol():
    """Test Tool protocol implementation from API reference."""

    try:

        class CalculatorArgs(BaseModel):
            expression: str = Field(description="Math expression to evaluate")

        class CalculatorTool:
            @property
            def schema(self):
                return type(
                    "ToolSchema",
                    (),
                    {
                        "name": "calculate",
                        "description": "Perform mathematical calculations",
                        "parameters": CalculatorArgs,
                    },
                )()

            async def execute(self, args: CalculatorArgs, context) -> str:
                result = eval(args.expression)  # Use safe evaluator in production
                return f"Result: {result}"

        tool = CalculatorTool()
        assert tool.schema.name == "calculate"
        assert tool.schema.description == "Perform mathematical calculations"
        assert tool.schema.parameters == CalculatorArgs
        results.success("api-reference: Tool protocol implementation")
    except Exception as e:
        results.failure("api-reference: Tool protocol implementation", str(e))


def test_agent_creation():
    """Test Agent creation from API reference."""

    try:

        @dataclass
        class TestContext:
            user_id: str
            permissions: List[str]

        # Create a simple tool for the agent
        class SimpleArgs(BaseModel):
            message: str = Field(description="Message to process")

        simple_tool = create_function_tool(
            {
                "name": "simple_tool",
                "description": "A simple test tool",
                "execute": lambda args, context: f"Processed: {args.message}",
                "parameters": SimpleArgs,
                "source": ToolSource.NATIVE,
            }
        )

        def create_assistant(context_type):
            def instructions(state: RunState[context_type]) -> str:
                return f"You are a helpful assistant. User: {state.context.user_id}"

            return Agent(
                name="Assistant",
                instructions=instructions,
                tools=[simple_tool],
                handoffs=["SpecialistAgent"],
            )

        agent = create_assistant(TestContext)
        assert agent.name == "Assistant"
        assert len(agent.tools) == 1
        assert agent.handoffs == ["SpecialistAgent"]

        # Test instructions function
        test_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[],
            current_agent_name="Assistant",
            context=TestContext(user_id="test_user", permissions=["read"]),
            turn_count=0,
        )

        instructions = agent.instructions(test_state)
        assert "test_user" in instructions
        results.success("api-reference: Agent creation")
    except Exception as e:
        results.failure("api-reference: Agent creation", str(e))


def test_tool_response_system():
    """Test ToolResponse helper system from API reference."""

    try:
        # Test success response
        success_result = ToolResponse.success(data={"result": 42}, metadata={"execution_time": 0.1})
        assert success_result.status == "success"
        assert success_result.data == {"result": 42}
        results.success("api-reference: ToolResponse.success")
    except Exception as e:
        results.failure("api-reference: ToolResponse.success", str(e))

    try:
        # Test error response
        error_result = ToolResponse.error(
            code="calculation_error", message="Division by zero", details={"expression": "1/0"}
        )
        assert error_result.status == "error"
        assert error_result.error.code == "calculation_error"
        results.success("api-reference: ToolResponse.error")
    except Exception as e:
        results.failure("api-reference: ToolResponse.error", str(e))

    try:
        # Test validation error
        validation_result = ToolResponse.validation_error(
            message="Invalid input format", details={"expected": "number", "received": "string"}
        )
        assert validation_result.status == "validation_error"
        results.success("api-reference: ToolResponse.validation_error")
    except Exception as e:
        results.failure("api-reference: ToolResponse.validation_error", str(e))

    try:
        # Test permission denied
        permission_result = ToolResponse.permission_denied(
            message="Access denied", required_permissions=["admin"]
        )
        assert permission_result.status == "permission_denied"
        results.success("api-reference: ToolResponse.permission_denied")
    except Exception as e:
        results.failure("api-reference: ToolResponse.permission_denied", str(e))


async def test_complete_example():
    """Test the complete example from API reference."""

    try:

        @dataclass
        class UserContext:
            user_id: str
            permissions: List[str]

        class CalculateArgs(BaseModel):
            expression: str = Field(description="Math expression to evaluate")

        class CalculatorTool:
            @property
            def schema(self):
                return type(
                    "ToolSchema",
                    (),
                    {
                        "name": "calculate",
                        "description": "Perform safe mathematical calculations",
                        "parameters": CalculateArgs,
                    },
                )()

            async def execute(self, args: CalculateArgs, context: UserContext) -> str:
                if "calculator" not in context.permissions:
                    return tool_result_to_string(
                        ToolResponse.permission_denied(
                            "Calculator access denied", required_permissions=["calculator"]
                        )
                    )

                try:
                    # Simple evaluation for testing
                    result = eval(args.expression)
                    return tool_result_to_string(
                        ToolResponse.success(f"Result: {args.expression} = {result}")
                    )
                except Exception as e:
                    return tool_result_to_string(ToolResponse.error("calculation_error", str(e)))

        # Test tool creation and basic functionality
        calculator = CalculatorTool()
        assert calculator.schema.name == "calculate"

        # Test with valid context
        context = UserContext(user_id="test", permissions=["calculator"])
        args = CalculateArgs(expression="2 + 3")

        result = await calculator.execute(args, context)
        assert "Result: 2 + 3 = 5" in result

        results.success("api-reference: complete example")
    except Exception as e:
        results.failure("api-reference: complete example", str(e))


async def main():
    """Run all API reference code tests."""
    print("🧪 Testing all code examples from docs/api-reference.md...")
    print("=" * 60)

    # Test enums
    print("\n📋 Testing enum definitions...")
    test_enums()

    # Test tool creation
    print("\n🔧 Testing tool creation...")
    test_tool_creation()

    # Test ID generation
    print("\n🆔 Testing ID generation...")
    test_id_generation()

    # Test core types
    print("\n📦 Testing core types...")
    test_core_types()

    # Test tool protocol
    print("\n🔌 Testing tool protocol...")
    test_tool_protocol()

    # Test agent creation
    print("\n🤖 Testing agent creation...")
    test_agent_creation()

    # Test tool response system
    print("\n📤 Testing tool response system...")
    test_tool_response_system()

    # Test complete example
    print("\n💡 Testing complete example...")
    await test_complete_example()

    # Print summary
    print("\n" + "=" * 60)
    results.summary()

    if results.failed > 0:
        print(f"\n⚠️  {results.failed} tests failed. API reference needs fixes.")
        return False
    else:
        print(f"\n🎉 All {results.passed} tests passed! API reference is accurate.")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
