import asyncio
import os
import sys
import operator as op
import ast
import socket
from typing import Any, Dict, Union, List
from pydantic import BaseModel, Field
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, '.')

from jaf.core.tools import create_function_tool
from jaf.core.types import Agent, Message, RunState, RunConfig, ModelConfig
from jaf.core.engine import run
from jaf.providers.model import make_litellm_provider

# JAF Tool Parameter Schema
class MathToolArgs(BaseModel):
    expression: Union[str, List[str]] = Field(
        description="Mathematical expression(s) to evaluate. Can be a single expression string or list of expressions."
    )
    variables: Dict[str, Union[float, List[float]]] = Field(
        default={},
        description="Dictionary of variables to use in the expression evaluation"
    )
    precision: Union[int, None] = Field(
        default=None,
        description="Number of decimal places to round the result to (optional)"
    )

# Safe allowed functions
SAFE_FUNCTIONS = {
    "sum": sum, "min": min, "max": max, "len": len, "abs": abs, "round": round,
    "mean": lambda x: sum(x) / len(x) if x else 0,
}

SAFE_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}

class SafeEvaluator(ast.NodeVisitor):
    def __init__(self, variables: Dict[str, Any]):
        self.variables = variables

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.BinOp):
            return SAFE_OPERATORS[type(node.op)](self.visit(node.left), self.visit(node.right))
        elif isinstance(node, ast.UnaryOp):
            return SAFE_OPERATORS[type(node.op)](self.visit(node.operand))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid constant type")
        elif isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            raise ValueError(f"Unknown variable '{node.id}'")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function call")
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' not allowed")
            args = [self.visit(arg) for arg in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
        elif isinstance(node, ast.List):
            return [self.visit(elt) for elt in node.elts]
        else:
            raise ValueError("Unsupported expression")

def safe_eval(expression: str, variables: Dict[str, Any]) -> Any:
    try:
        node = ast.parse(expression, mode="eval")
        evaluator = SafeEvaluator(variables)
        return evaluator.visit(node)
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

# JAF Tool Implementation
async def math_execute(args: MathToolArgs, context: Any = None) -> str:
    if isinstance(args.expression, list):
        result = []
        for expr in args.expression:
            res = safe_eval(expr, args.variables)
            if args.precision is not None:
                res = round(res, args.precision)
            result.append(res)
        return f"MAGIC_MATH_TOOL_RESULT: {result}"
    
    result = safe_eval(args.expression, args.variables)
    if args.precision is not None:
        result = round(result, args.precision)
    return f"MAGIC_MATH_TOOL_RESULT: {result}"

# JAF Tool Definition
math_tool_description = """
    Evaluates one or more mathematical expression safely. Properly understand and analyze the question, metrics on which the calculation needs to be performed, what operations to be performed for getting an accurate and precise answer., 
    Args:
        expression (str | List(str)): The mathematical expression / list of expressions to evaluate.
        variables (dict, optional): A dictionary of variable names and their values.
        precision (int, optional): Number of decimal places to round the result.
    Returns:
        dict: A dictionary containing the result of the evaluation.
    """

math_tool_jaf = create_function_tool({
    'name': 'math',
    'description': math_tool_description,
    'execute': math_execute,
    'parameters': MathToolArgs,
    'metadata': {'category': 'computation', 'priority': 'medium'}
})

def math_agent_instructions(state):
    return 'You are a math assistant with a calculator tool.'

def check_litellm_available():
    """Check if LiteLLM server is available."""
    try:
        litellm_url = os.getenv("LITELLM_URL", "http://0.0.0.0:4000/")
        # Parse URL to get host and port
        from urllib.parse import urlparse
        parsed = urlparse(litellm_url)
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 4000
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False

def check_env_available():
    """Check if required environment variables are available."""
    litellm_url = os.getenv("LITELLM_URL")
    # Either LITELLM_URL should be set, or we should have some API key
    return litellm_url is not None or os.getenv("LITELLM_API_KEY") is not None

skip_if_no_litellm = pytest.mark.skipif(
    not check_litellm_available() or not check_env_available(),
    reason="Skipping math tool test: LiteLLM server not available or environment variables (LITELLM_URL/LITELLM_API_KEY) not set. Please start LiteLLM server and configure .env file to run this test."
)

@skip_if_no_litellm
async def test_math_tool_integration():
    """Test that the math_tool_jaf works with the JAF engine using real LiteLLM."""
    
    print("🧪 Testing math_tool_jaf integration with JAF engine using real LiteLLM...")
    print("=" * 65)
    
    agent = Agent(
        name='AdvancedMathAgent',
        instructions=math_agent_instructions,
        tools=[math_tool_jaf],
        model_config=ModelConfig(name="gemini-2.5-pro")
    )
    
    print(f"✅ Agent created with tool: {agent.tools[0].schema.name}")
    
    # Use real LiteLLM provider
    litellm_url = os.getenv("LITELLM_URL", "http://0.0.0.0:4000/")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "")
    
    model_provider = make_litellm_provider(
        base_url=litellm_url,
        api_key=litellm_api_key
    )
    
    print(f"✅ Using real LiteLLM provider: {litellm_url}")
    
    initial_state = RunState(
        run_id='run-math-123',
        trace_id='trace-math-456',
        messages=[Message(role='user', content='What is (100 / 5) + 2?')],
        current_agent_name='AdvancedMathAgent',
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={'AdvancedMathAgent': agent},
        model_provider=model_provider,
        max_turns=3
    )
    
    print("🚀 Running JAF engine with math tool...")
    
    try:
        result = await run(initial_state, config)
        print(f"✅ Engine run completed successfully")
        
        print(f"DEBUG: Final Messages = {result.final_state.messages}")
        
        tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
        assert len(tool_messages) > 0, "No tool messages found in the final state."
        assert "MAGIC_MATH_TOOL_RESULT: 22.0" in tool_messages[0].content
        
        print(f"✅ Tool execution successful!")
        print(f"   Tool result: {tool_messages[0].content}")
        
    except Exception as e:
        print(f"❌ Engine run failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 65)
    print("🎯 Math Tool Integration Test Passed!")

if __name__ == "__main__":
    asyncio.run(test_math_tool_integration())
