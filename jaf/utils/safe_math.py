"""Safe arithmetic evaluation for calculator-style tools (never use eval() on model input)."""

import ast
import operator
from typing import Callable, Mapping, Optional, Union

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

Number = Union[int, float]


def safe_eval_arithmetic(
    expression: str,
    functions: Optional[Mapping[str, Callable[..., Number]]] = None,
    constants: Optional[Mapping[str, Number]] = None,
) -> Number:
    """Evaluate +, -, *, /, //, % over numbers; anything else raises ValueError.

    ``functions`` and ``constants`` are explicit allowlists of callable and named values
    (e.g. ``{"sqrt": math.sqrt}`` / ``{"pi": math.pi}``).
    """
    functions = functions or {}
    constants = constants or {}

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and not node.keywords
        ):
            return functions[node.func.id](*[_eval(arg) for arg in node.args])
        raise ValueError("Unsupported expression")

    try:
        return _eval(ast.parse(expression.strip(), mode="eval"))
    except SyntaxError as e:
        raise ValueError("Invalid expression") from e
