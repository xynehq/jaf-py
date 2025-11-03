"""
ADK Safe Mathematical Expression Evaluator

This module provides a secure alternative to eval() for mathematical expressions,
preventing code injection vulnerabilities while supporting basic arithmetic operations.
"""

import ast
import operator
from typing import Union, Any


class SafeMathEvaluator:
    """
    Safe mathematical expression evaluator without code injection risk.

    This evaluator parses mathematical expressions into Abstract Syntax Trees (AST)
    and evaluates them using a whitelist of safe operations, preventing any
    possibility of code injection attacks.
    """

    # Whitelist of safe mathematical operations
    SAFE_OPERATORS = {
        ast.Add: operator.add,  # +
        ast.Sub: operator.sub,  # -
        ast.Mult: operator.mul,  # *
        ast.Div: operator.truediv,  # /
        ast.FloorDiv: operator.floordiv,  # //
        ast.Mod: operator.mod,  # %
        ast.Pow: operator.pow,  # **
        ast.USub: operator.neg,  # -x (unary minus)
        ast.UAdd: operator.pos,  # +x (unary plus)
    }

    # Whitelist of safe mathematical functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    @classmethod
    def safe_eval(cls, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string (e.g., "2 + 3 * 4")

        Returns:
            Numerical result of the expression

        Raises:
            ValueError: If the expression contains unsafe operations or syntax errors
            ZeroDivisionError: If division by zero occurs
            TypeError: If invalid types are used in operations
        """
        if not expression or not expression.strip():
            raise ValueError("Empty expression")

        # Basic validation: only allow safe characters
        allowed_chars = set("0123456789+-*/.() abcdefghijklmnopqrstuvwxyz")
        if not all(c.lower() in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")

        # Parse the expression into an AST
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid mathematical syntax: {e}")

        # Evaluate the AST safely
        return cls._eval_node(tree.body)

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        """
        Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate

        Returns:
            Numerical result of the node evaluation

        Raises:
            ValueError: If the node contains unsafe operations
        """
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.BinOp):  # Binary operations (e.g., 1 + 2)
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)

            if type(node.op) not in cls.SAFE_OPERATORS:
                raise ValueError(f"Unsupported binary operation: {type(node.op).__name__}")

            operation = cls.SAFE_OPERATORS[type(node.op)]

            try:
                return operation(left, right)
            except ZeroDivisionError:
                raise ZeroDivisionError("Division by zero")
            except Exception as e:
                raise ValueError(f"Operation error: {e}")

        elif isinstance(node, ast.UnaryOp):  # Unary operations (e.g., -5)
            operand = cls._eval_node(node.operand)

            if type(node.op) not in cls.SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")

            operation = cls.SAFE_OPERATORS[type(node.op)]
            return operation(operand)

        elif isinstance(node, ast.Call):  # Function calls (e.g., abs(-5))
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")

            func_name = node.func.id
            if func_name not in cls.SAFE_FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")

            # Evaluate arguments
            args = [cls._eval_node(arg) for arg in node.args]

            # Check for keyword arguments (not supported for security)
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")

            try:
                return cls.SAFE_FUNCTIONS[func_name](*args)
            except Exception as e:
                raise ValueError(f"Function call error: {e}")

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_calculate(expression: str) -> dict[str, Any]:
    """
    Safely calculate a mathematical expression and return a structured result.

    This function is a convenient wrapper around SafeMathEvaluator that returns
    a dictionary with the result or error information, suitable for API responses.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Dictionary containing either 'result' or 'error' key
    """
    from datetime import datetime

    try:
        result = SafeMathEvaluator.safe_eval(expression)
        return {
            "result": result,
            "expression": expression,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }
    except (ValueError, ZeroDivisionError, TypeError) as e:
        return {
            "error": str(e),
            "expression": expression,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "expression": expression,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
        }


# Example usage and testing
if __name__ == "__main__":
    # Test cases for the safe evaluator
    test_expressions = [
        "2 + 3",
        "10 * (5 + 3)",
        "100 / 4",
        "2 ** 3",
        "-5 + 10",
        "abs(-42)",
        "max(10, 20, 5)",
        "round(3.14159, 2)",
        # These should fail safely:
        # "import os",
        # "__import__('os')",
        # "eval('1+1')",
    ]

    print("🧪 Testing SafeMathEvaluator:")
    print("=" * 40)

    for expr in test_expressions:
        result = safe_calculate(expr)
        if result["status"] == "success":
            print(f"✅ {expr} = {result['result']}")
        else:
            print(f"❌ {expr} -> {result['error']}")
