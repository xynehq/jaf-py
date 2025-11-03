#!/usr/bin/env python3
"""
Comprehensive test suite for all code examples in JAF documentation.
This ensures every code snippet in the docs actually works with the real implementation.
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

# Test imports from the documentation examples
try:
    from pydantic import BaseModel, Field, field_validator
    from jaf import create_function_tool, ToolSource
    from jaf import ToolResponse, ToolResult
    from jaf import Agent
    from jaf.core.types import RunState, RunConfig, Message, generate_run_id, generate_trace_id
    from adk.runners import run_agent
    from jaf import make_litellm_provider
    from jaf.core.composition import (
        create_tool_pipeline,
        create_parallel_tools,
        create_conditional_tool,
        with_retry,
        with_cache,
        with_timeout,
        compose,
    )

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class DocumentationResultTracker:
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


results = DocumentationResultTracker()


# Test core-concepts.md examples
def test_core_concepts_examples():
    """Test code examples from docs/core-concepts.md"""

    # Test immutable state example
    try:
        from dataclasses import replace

        @dataclass(frozen=True)
        class TestState:
            messages: List[str]
            count: int

        state = TestState(messages=["hello"], count=1)
        new_state = replace(state, messages=[*state.messages, "world"], count=state.count + 1)

        assert state.messages == ["hello"]
        assert new_state.messages == ["hello", "world"]
        assert new_state.count == 2
        results.success("core-concepts: immutable state pattern")
    except Exception as e:
        results.failure("core-concepts: immutable state pattern", str(e))

    # Test context definition
    try:

        @dataclass
        class ECommerceContext:
            user_id: str
            cart_items: List[str]
            is_premium: bool

        context = ECommerceContext(user_id="user123", cart_items=[], is_premium=True)
        assert context.user_id == "user123"
        assert context.is_premium is True
        results.success("core-concepts: context definition")
    except Exception as e:
        results.failure("core-concepts: context definition", str(e))


# Test tools.md examples
def test_tools_examples():
    """Test code examples from docs/tools.md"""

    # Test modern object-based API
    try:

        class MyToolArgs(BaseModel):
            param1: str = Field(description="Description of parameter")
            param2: int = Field(default=0, description="Optional parameter with default")

        async def my_tool_execute(args: MyToolArgs, context: Any) -> ToolResult[str]:
            return ToolResponse.success(f"Processed {args.param1} with {args.param2}")

        my_tool = create_function_tool(
            {
                "name": "my_tool",
                "description": "What this tool does",
                "execute": my_tool_execute,
                "parameters": MyToolArgs,
                "metadata": {"category": "utility"},
                "source": ToolSource.NATIVE,
            }
        )

        assert my_tool.schema.name == "my_tool"
        assert my_tool.schema.description == "What this tool does"
        results.success("tools: modern object-based API")
    except Exception as e:
        results.failure("tools: modern object-based API", str(e))

    # Test Pydantic V2 field validator
    try:

        class ValidatedArgs(BaseModel):
            expression: str = Field(description="Mathematical expression")

            @field_validator("expression")
            @classmethod
            def validate_expression_safety(cls, v):
                allowed_chars = set("0123456789+-*/(). ")
                if not all(c in allowed_chars for c in v):
                    raise ValueError("Expression contains invalid characters")
                return v

        # Test valid expression
        valid_args = ValidatedArgs(expression="2 + 2")
        assert valid_args.expression == "2 + 2"

        # Test invalid expression should raise error
        try:
            ValidatedArgs(expression="import os")
            results.failure("tools: Pydantic V2 validator", "Should have raised validation error")
        except ValueError:
            results.success("tools: Pydantic V2 field validator")
    except Exception as e:
        results.failure("tools: Pydantic V2 field validator", str(e))


# Test function-composition.md examples
def test_composition_examples():
    """Test code examples from docs/function-composition.md"""

    # Test basic tool composition
    try:
        # Create a simple tool for testing
        class TestArgs(BaseModel):
            value: str

        async def test_execute(args: TestArgs, context):
            return ToolResponse.success(f"Processed: {args.value}")

        base_tool = create_function_tool(
            {
                "name": "test_tool",
                "description": "Test tool",
                "execute": test_execute,
                "parameters": TestArgs,
                "source": ToolSource.NATIVE,
            }
        )

        # Test caching composition
        cached_tool = with_cache(base_tool, ttl_seconds=60)
        assert cached_tool.schema.name == "test_tool_cached"
        results.success("composition: with_cache")

        # Test retry composition
        retry_tool = with_retry(base_tool, max_retries=3)
        assert retry_tool.schema.name == "test_tool_retry"
        results.success("composition: with_retry")

        # Test timeout composition
        timeout_tool = with_timeout(base_tool, timeout_seconds=30)
        assert timeout_tool.schema.name == "test_tool_timeout"
        results.success("composition: with_timeout")

    except Exception as e:
        results.failure("composition: basic tool composition", str(e))

    # Test tool pipeline
    try:

        class SimpleArgs(BaseModel):
            data: str

        async def step1_execute(args: SimpleArgs, context):
            return ToolResponse.success(f"step1: {args.data}")

        async def step2_execute(args: Any, context):
            return ToolResponse.success(f"step2: processed")

        tool1 = create_function_tool(
            {
                "name": "step1",
                "description": "First step",
                "execute": step1_execute,
                "parameters": SimpleArgs,
                "source": ToolSource.NATIVE,
            }
        )

        tool2 = create_function_tool(
            {
                "name": "step2",
                "description": "Second step",
                "execute": step2_execute,
                "parameters": SimpleArgs,
                "source": ToolSource.NATIVE,
            }
        )

        pipeline_tool = create_tool_pipeline(tool1, tool2, name="test_pipeline")
        assert pipeline_tool.schema.name == "test_pipeline_pipeline"
        results.success("composition: tool pipeline")

    except Exception as e:
        results.failure("composition: tool pipeline", str(e))

    # Test parallel tools
    try:
        parallel_tool = create_parallel_tools(tool1, tool2, name="test_parallel")
        assert parallel_tool.schema.name == "test_parallel_parallel"
        results.success("composition: parallel tools")

    except Exception as e:
        results.failure("composition: parallel tools", str(e))

    # Test conditional tool
    try:

        def simple_condition(args):
            return hasattr(args, "data") and "test" in args.data

        conditional_tool = create_conditional_tool(
            simple_condition, tool1, tool2, name="test_conditional"
        )
        assert conditional_tool.schema.name == "test_conditional_conditional"
        results.success("composition: conditional tool")

    except Exception as e:
        results.failure("composition: conditional tool", str(e))

    # Test composition builder
    try:
        composed_tool = (
            compose(base_tool).with_cache(ttl_seconds=300).with_retry(max_retries=2).build()
        )
        assert composed_tool is not None
        results.success("composition: builder pattern")

    except Exception as e:
        results.failure("composition: builder pattern", str(e))


# Test getting-started.md examples (more comprehensive)
async def test_getting_started_examples():
    """Test code examples from docs/getting-started.md"""

    # Test calculator context
    try:

        @dataclass(frozen=True)
        class CalculatorContext:
            user_id: str
            session_id: str
            allowed_operations: List[str]
            max_result: float = 1000000.0
            precision: int = 10
            user_permissions: List[str] = None
            session_metadata: Optional[Dict[str, Any]] = None
            created_at: datetime = None

            def __post_init__(self):
                if self.user_permissions is None:
                    object.__setattr__(self, "user_permissions", ["basic_math"])
                if self.created_at is None:
                    object.__setattr__(self, "created_at", datetime.now(datetime.UTC))

            def has_permission(self, operation: str) -> bool:
                return operation in self.user_permissions

            def can_perform_operation(self, operation: str) -> bool:
                return operation in self.allowed_operations

        context = CalculatorContext(
            user_id="test_user", session_id="test_session", allowed_operations=["add", "multiply"]
        )

        assert context.has_permission("basic_math")
        assert context.can_perform_operation("add")
        assert not context.can_perform_operation("divide")
        results.success("getting-started: calculator context")

    except Exception as e:
        results.failure("getting-started: calculator context", str(e))

    # Test calculator args with Pydantic V2
    try:

        class CalculateArgs(BaseModel):
            expression: str = Field(
                description="Mathematical expression to evaluate", min_length=1, max_length=200
            )

            @field_validator("expression")
            @classmethod
            def validate_expression_safety(cls, v):
                cleaned = v.replace(" ", "")
                dangerous_patterns = ["__", "import", "exec", "eval"]
                for pattern in dangerous_patterns:
                    if pattern in cleaned.lower():
                        raise ValueError(f"Expression contains prohibited pattern: {pattern}")
                return v

        args = CalculateArgs(expression="2 + 2")
        assert args.expression == "2 + 2"
        results.success("getting-started: calculator args with validation")

    except Exception as e:
        results.failure("getting-started: calculator args with validation", str(e))

    # Test tool creation and execution
    try:

        async def calculate_execute(
            args: CalculateArgs, context: CalculatorContext
        ) -> ToolResult[str]:
            if not context.has_permission("basic_math"):
                return ToolResponse.permission_denied(
                    "Mathematical operations require basic_math permission",
                    required_permissions=["basic_math"],
                )

            try:
                # Simple evaluation for testing
                allowed_chars = set("0123456789+-*/.() ")
                if not all(char in allowed_chars for char in args.expression):
                    return ToolResponse.validation_error("Expression contains invalid characters")

                result = eval(args.expression)

                if abs(result) > context.max_result:
                    return ToolResponse.validation_error(
                        f"Result {result} exceeds maximum allowed value ({context.max_result})"
                    )

                if isinstance(result, float):
                    result = round(result, context.precision)

                return ToolResponse.success(
                    data=f"Result: {args.expression} = {result}",
                    metadata={
                        "operation_count": args.expression.count("+") + args.expression.count("-"),
                        "result_type": type(result).__name__,
                        "precision_used": context.precision,
                    },
                )

            except Exception as e:
                return ToolResponse.error(
                    code="calculation_error",
                    message=f"Failed to evaluate expression: {str(e)}",
                    details={"expression": args.expression, "error_type": type(e).__name__},
                )

        calculator_tool = create_function_tool(
            {
                "name": "calculate",
                "description": "Safely evaluate mathematical expressions",
                "execute": calculate_execute,
                "parameters": CalculateArgs,
                "metadata": {
                    "category": "mathematical_operations",
                    "safety_level": "high",
                    "version": "1.0.0",
                },
                "source": ToolSource.NATIVE,
            }
        )

        # Test tool execution
        test_context = CalculatorContext(
            user_id="test", session_id="test", allowed_operations=["add"]
        )
        test_args = CalculateArgs(expression="2 + 3")

        result = await calculator_tool.execute(test_args, test_context)
        assert result.status == "success"
        assert "Result: 2 + 3 = 5" in result.data
        results.success("getting-started: calculator tool execution")

    except Exception as e:
        results.failure("getting-started: calculator tool execution", str(e))

    # Test agent creation
    try:

        def create_calculator_agent() -> Agent[CalculatorContext, str]:
            def instructions(state: RunState[CalculatorContext]) -> str:
                calc_count = len([m for m in state.messages if "calculate" in m.content.lower()])

                base_instruction = """You are a helpful calculator assistant. You can perform mathematical calculations safely.
                
Available operations: addition (+), subtraction (-), multiplication (*), division (/), parentheses ()

Rules:
- Always use the calculate tool for mathematical expressions
- Explain your calculations clearly
- Results are limited to values under 1,000,000"""

                if calc_count > 3:
                    base_instruction += "\n\nNote: You've performed several calculations. Consider summarizing results if helpful."

                return base_instruction

            return Agent(name="Calculator", instructions=instructions, tools=[calculator_tool])

        agent = create_calculator_agent()
        assert agent.name == "Calculator"
        assert len(agent.tools) == 1
        assert agent.tools[0].schema.name == "calculate"

        # Test that instructions function works
        mock_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role="user", content="test calculate something")],
            current_agent_name="Calculator",
            context=CalculatorContext(
                user_id="test", session_id="test", allowed_operations=["add"]
            ),
            turn_count=0,
        )
        instructions = agent.instructions(mock_state)
        assert "calculator assistant" in instructions.lower()
        results.success("getting-started: agent creation")

    except Exception as e:
        results.failure("getting-started: agent creation", str(e))

    # Test state creation
    try:
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role="user", content="What is 15 * 8 + 32?")],
            current_agent_name="Calculator",
            context=CalculatorContext(
                user_id="demo_user",
                session_id="demo_session",
                allowed_operations=["add", "subtract", "multiply", "divide"],
            ),
            turn_count=0,
        )

        assert initial_state.run_id is not None
        assert initial_state.trace_id is not None
        assert len(initial_state.messages) == 1
        assert initial_state.current_agent_name == "Calculator"
        results.success("getting-started: state creation")

    except Exception as e:
        results.failure("getting-started: state creation", str(e))


# Test examples.md code snippets
def test_examples_code():
    """Test code examples from docs/examples.md"""

    # Test context definition from examples
    try:

        @dataclass
        class MyContext:
            user_id: str
            permissions: List[str]

        context = MyContext(user_id="test", permissions=["user"])
        assert context.user_id == "test"
        assert "user" in context.permissions
        results.success("examples: context definition")

    except Exception as e:
        results.failure("examples: context definition", str(e))

    # Test greeting tool from examples
    try:

        class GreetArgs(BaseModel):
            name: str = Field(description="Name to greet")
            style: str = Field(default="friendly", description="Greeting style")

        class GreetingTool:
            @property
            def schema(self):
                return type(
                    "ToolSchema",
                    (),
                    {
                        "name": "greet",
                        "description": "Generate a personalized greeting",
                        "parameters": GreetArgs,
                    },
                )()

            async def execute(self, args: GreetArgs, context) -> Any:
                if not args.name or args.name.strip() == "":
                    return ToolResponse.validation_error(
                        "Name cannot be empty", {"provided_name": args.name}
                    )

                if len(args.name) > 100:
                    return ToolResponse.validation_error(
                        "Name is too long (max 100 characters)",
                        {"name_length": len(args.name), "max_length": 100},
                    )

                greeting = f"Hello, {args.name.strip()}! Nice to meet you."

                return ToolResponse.success(
                    greeting, {"greeted_name": args.name.strip(), "greeting_type": "personal"}
                )

        tool = GreetingTool()
        assert tool.schema.name == "greet"
        results.success("examples: greeting tool structure")

    except Exception as e:
        results.failure("examples: greeting tool structure", str(e))


async def main():
    """Run all documentation code tests."""
    print("🧪 Testing all code examples from JAF documentation...")
    print("=" * 60)

    # Test core concepts
    print("\n📖 Testing core-concepts.md examples...")
    test_core_concepts_examples()

    # Test tools guide
    print("\n🔧 Testing tools.md examples...")
    test_tools_examples()

    # Test function composition
    print("\n🔗 Testing function-composition.md examples...")
    test_composition_examples()

    # Test getting started guide
    print("\n🚀 Testing getting-started.md examples...")
    await test_getting_started_examples()

    # Test examples guide
    print("\n💡 Testing examples.md code snippets...")
    test_examples_code()

    # Print summary
    print("\n" + "=" * 60)
    results.summary()

    if results.failed > 0:
        print(f"\n⚠️  {results.failed} tests failed. Documentation needs fixes.")
        return False
    else:
        print(f"\n🎉 All {results.passed} tests passed! Documentation is accurate.")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
