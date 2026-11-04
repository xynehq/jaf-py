"""print() is banned in the jaf package: it bypasses log levels and has leaked PII to stdout.

Use ``logger = logging.getLogger(__name__)`` instead. Ruff enforces the same rule (T201/T203).
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "jaf"
BANNED = {"print", "pprint"}


def _banned_calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in BANNED:
            yield node.lineno, func.id
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "pprint"
        ):
            yield node.lineno, f"pprint.{func.attr}"


def test_no_print_calls_in_jaf_package():
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT.parent)}:{lineno}: {name}()"
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for lineno, name in _banned_calls(path)
    ]
    assert not offenders, "print() is banned in jaf/, use logging:\n" + "\n".join(offenders)
