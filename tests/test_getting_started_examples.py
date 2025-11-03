#!/usr/bin/env python3
"""
Test script to verify all code examples from docs/getting-started.md work correctly.
This will help identify any discrepancies between documentation and actual implementation.
"""

import pytest
import asyncio
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

# Test imports from the documentation examples
from pydantic import BaseModel, Field, field_validator
from jaf import create_function_tool, ToolSource
from jaf import ToolResponse, ToolResult
from jaf import Agent
from jaf.core.types import RunState, RunConfig, Message, generate_run_id, generate_trace_id
from adk.runners import run_agent
from jaf import make_litellm_provider


# Test the CalculatorContext from the documentation
@dataclass(frozen=True)
class CalculatorContext:
    """
    Immutable context for calculator agent operations.
    """

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
            object.__setattr__(self, "created_at", datetime.now(timezone.utc))

    def has_permission(self, operation: str) -> bool:
        """Check if user has permission for specific operation."""
        return operation in self.user_permissions

    def can_perform_operation(self, operation: str) -> bool:
        """Check if operation is allowed in current context."""
        return operation in self.allowed_operations


# Test the CalculateArgs from the documentation
class CalculateArgs(BaseModel):
    """
    Arguments for mathematical calculation tool.
    """

    expression: str = Field(
        description="Mathematical expression to evaluate (e.g., '2 + 2', '(10 * 5) / 2')",
        min_length=1,
        max_length=200,
    )

    @field_validator("expression")
    @classmethod
    def validate_expression_safety(cls, v):
        """Ensure expression contains only safe mathematical operations."""
        # Remove whitespace for validation
        cleaned = v.replace(" ", "")

        # Check for potentially dangerous patterns
        dangerous_patterns = [
            "__",
            "import",
            "exec",
            "eval",
            "open",
            "file",
            "input",
            "raw_input",
            "compile",
            "globals",
            "locals",
        ]

        for pattern in dangerous_patterns:
            if pattern in cleaned.lower():
                raise ValueError(f"Expression contains prohibited pattern: {pattern}")

        return v


# Test the calculate_execute function from the documentation
async def calculate_execute(args: CalculateArgs, context: CalculatorContext) -> ToolResult[str]:
    """
    Execute mathematical calculation with comprehensive safety checks.
    """
    try:
        # Permission check
        if not context.has_permission("basic_math"):
            return ToolResponse.permission_denied(
                "Mathematical operations require basic_math permission",
                required_permissions=["basic_math"],
            )

        # Simple evaluation for testing (in production, use AST parsing)
        try:
            # Basic whitelist validation
            allowed_chars = set("0123456789+-*/.() ")
            if not all(char in allowed_chars for char in args.expression):
                return ToolResponse.validation_error("Expression contains invalid characters")

            result = eval(args.expression)
        except (SyntaxError, ValueError) as e:
            return ToolResponse.validation_error(f"Invalid mathematical expression: {str(e)}")

        # Apply context limits
        if abs(result) > context.max_result:
            return ToolResponse.validation_error(
                f"Result {result} exceeds maximum allowed value ({context.max_result})"
            )

        # Format result with context precision
        if isinstance(result, float):
            result = round(result, context.precision)

        return ToolResponse.success(
            data=f"Result: {args.expression} = {result}",
            metadata={
                "operation_count": args.expression.count("+")
                + args.expression.count("-")
                + args.expression.count("*")
                + args.expression.count("/"),
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


# Test tool creation from the documentation
@pytest.fixture
def calculator_tool():
    """Test creating the calculator tool as shown in documentation."""
    calculator_tool = create_function_tool(
        {
            "name": "calculate",
            "description": "Safely evaluate mathematical expressions using AST parsing",
            "execute": calculate_execute,
            "parameters": CalculateArgs,
            "metadata": {
                "category": "mathematical_operations",
                "safety_level": "high",
                "supported_operations": ["addition", "subtraction", "multiplication", "division"],
                "security_features": ["ast_parsing", "operation_whitelisting", "result_validation"],
                "version": "1.0.0",
            },
            "source": ToolSource.NATIVE,
        }
    )
    return calculator_tool


def test_tool_creation():
    """Test creating the calculator tool as shown in documentation."""
    calculator_tool = create_function_tool(
        {
            "name": "calculate",
            "description": "Safely evaluate mathematical expressions using AST parsing",
            "execute": calculate_execute,
            "parameters": CalculateArgs,
            "metadata": {
                "category": "mathematical_operations",
                "safety_level": "high",
                "supported_operations": ["addition", "subtraction", "multiplication", "division"],
                "security_features": ["ast_parsing", "operation_whitelisting", "result_validation"],
                "version": "1.0.0",
            },
            "source": ToolSource.NATIVE,
        }
    )

    assert calculator_tool is not None
    assert calculator_tool.schema.name == "calculate"
    assert (
        calculator_tool.schema.description
        == "Safely evaluate mathematical expressions using AST parsing"
    )


# Test agent creation from the documentation
def test_agent_creation(calculator_tool):
    """Test creating the calculator agent as shown in documentation."""

    def instructions(state):
        """Dynamic instructions based on current state."""
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

    calculator_agent = Agent(name="Calculator", instructions=instructions, tools=[calculator_tool])

    assert calculator_agent is not None
    assert calculator_agent.name == "Calculator"
    assert len(calculator_agent.tools) == 1


def test_agent_creation_standalone():
    """Test creating the calculator agent without fixture dependency."""
    # Create tool inline
    calculator_tool = create_function_tool(
        {
            "name": "calculate",
            "description": "Safely evaluate mathematical expressions using AST parsing",
            "execute": calculate_execute,
            "parameters": CalculateArgs,
            "metadata": {
                "category": "mathematical_operations",
                "safety_level": "high",
                "supported_operations": ["addition", "subtraction", "multiplication", "division"],
                "security_features": ["ast_parsing", "operation_whitelisting", "result_validation"],
                "version": "1.0.0",
            },
            "source": ToolSource.NATIVE,
        }
    )

    def instructions(state):
        """Dynamic instructions based on current state."""
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

    calculator_agent = Agent(name="Calculator", instructions=instructions, tools=[calculator_tool])

    assert calculator_agent is not None
    assert calculator_agent.name == "Calculator"
    assert len(calculator_agent.tools) == 1


# Test context creation
@pytest.fixture
def context():
    """Test creating the calculator context as shown in documentation."""
    context = CalculatorContext(
        user_id="demo_user",
        session_id="test_session",
        allowed_operations=["add", "subtract", "multiply", "divide"],
    )
    return context


def test_context_creation():
    """Test creating the calculator context as shown in documentation."""
    context = CalculatorContext(
        user_id="demo_user",
        session_id="test_session",
        allowed_operations=["add", "subtract", "multiply", "divide"],
    )

    assert context is not None
    assert context.user_id == "demo_user"
    assert context.session_id == "test_session"
    assert context.has_permission("basic_math")
    assert context.can_perform_operation("add")


# Test state creation
def test_state_creation(context):
    """Test creating run state as shown in documentation."""
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role="user", content="What is 15 * 8 + 32?")],
        current_agent_name="Calculator",
        context=context,
        turn_count=0,
    )

    assert initial_state is not None
    assert initial_state.current_agent_name == "Calculator"
    assert len(initial_state.messages) == 1
    assert initial_state.turn_count == 0
    assert initial_state.context == context


def test_state_creation_standalone():
    """Test creating run state without fixture dependency."""
    # Create context inline
    context = CalculatorContext(
        user_id="demo_user",
        session_id="test_session",
        allowed_operations=["add", "subtract", "multiply", "divide"],
    )

    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role="user", content="What is 15 * 8 + 32?")],
        current_agent_name="Calculator",
        context=context,
        turn_count=0,
    )

    assert initial_state is not None
    assert initial_state.current_agent_name == "Calculator"
    assert len(initial_state.messages) == 1
    assert initial_state.turn_count == 0
    assert initial_state.context == context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
