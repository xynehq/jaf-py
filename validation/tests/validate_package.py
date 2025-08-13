#!/usr/bin/env python3
"""
Package validation script for JAF Python.

This script validates that the Python package is complete and ready for production.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and report."""
    if path.exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False


def check_import(module: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module)
        print(f"✅ Import {module}: OK")
        return True
    except ImportError as e:
        print(f"❌ Import {module}: FAILED - {e}")
        return False


def validate_package_structure() -> bool:
    """Validate the package structure."""
    print("🔍 Validating package structure...")

    root = Path(__file__).parent.parent.parent  # Go up to project root
    all_good = True

    # Core files
    required_files = [
        (root / "jaf" / "__init__.py", "Main package init"),
        (root / "jaf" / "cli.py", "CLI module"),
        (root / "pyproject.toml", "Package configuration"),
        (root / "README.md", "Documentation"),

        # Core modules
        (root / "jaf" / "core" / "__init__.py", "Core init"),
        (root / "jaf" / "core" / "engine.py", "Core engine"),
        (root / "jaf" / "core" / "types.py", "Core types"),
        (root / "jaf" / "core" / "errors.py", "Error handling"),
        (root / "jaf" / "core" / "tool_results.py", "Tool results"),
        (root / "jaf" / "core" / "tracing.py", "Tracing"),

        # Providers
        (root / "jaf" / "providers" / "__init__.py", "Providers init"),
        (root / "jaf" / "providers" / "model.py", "Model provider"),
        (root / "jaf" / "providers" / "mcp.py", "MCP provider"),

        # Policies
        (root / "jaf" / "policies" / "__init__.py", "Policies init"),
        (root / "jaf" / "policies" / "validation.py", "Validation policies"),
        (root / "jaf" / "policies" / "handoff.py", "Handoff policies"),

        # Server
        (root / "jaf" / "server" / "__init__.py", "Server init"),
        (root / "jaf" / "server" / "server.py", "Server implementation"),
        (root / "jaf" / "server" / "types.py", "Server types"),

        # Examples
        (root / "examples" / "server_example.py", "Server example"),
        (root / "examples" / "iterative_search_agent.py", "Iterative search agent example"),

        # Tests
        (root / "tests" / "test_engine.py", "Engine tests"),
        (root / "tests" / "test_validation.py", "Validation tests"),
    ]

    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_good = False

    return all_good


def validate_imports() -> bool:
    """Validate that key imports work."""
    print("\n🔍 Validating imports...")

    all_good = True

    # Core imports
    imports_to_test = [
        "jaf",
        "jaf.core.engine",
        "jaf.core.types",
        "jaf.core.errors",
        "jaf.core.tool_results",
        "jaf.core.tracing",
        "jaf.providers.model",
        "jaf.providers.mcp",
        "jaf.policies.validation",
        "jaf.policies.handoff",
        "jaf.server.server",
        "jaf.cli",
    ]

    for module in imports_to_test:
        if not check_import(module):
            all_good = False

    return all_good


def validate_key_exports() -> bool:
    """Validate that key exports are available."""
    print("\n🔍 Validating key exports...")

    try:
        import jaf

        # Check key exports
        key_exports = [
            "Agent", "Tool", "RunState", "RunConfig", "run",
            "generate_trace_id", "generate_run_id",
            "make_litellm_provider", "run_server",
            "MCPClient", "create_mcp_stdio_client",
            "ToolResult", "ToolResultStatus"
        ]

        all_good = True
        for export in key_exports:
            if hasattr(jaf, export):
                print(f"✅ Export {export}: Available")
            else:
                print(f"❌ Export {export}: Missing")
                all_good = False

        return all_good

    except Exception as e:
        print(f"❌ Failed to validate exports: {e}")
        return False


def validate_dependencies() -> bool:
    """Validate that dependencies are correctly specified."""
    print("\n🔍 Validating dependencies...")

    try:
        import fastapi
        import httpx
        import openai
        import pydantic
        import uvicorn
        import websockets

        print("✅ Core dependencies: Available")

        # Check optional dependencies
        optional_deps = {
            "google-generativeai": "google.generativeai",
            "python-dotenv": "dotenv"
        }

        for dep_name, module_name in optional_deps.items():
            try:
                __import__(module_name)
                print(f"✅ Optional dependency {dep_name}: Available")
            except ImportError:
                print(f"ℹ️  Optional dependency {dep_name}: Not installed (OK)")

        return True

    except ImportError as e:
        print(f"❌ Missing core dependency: {e}")
        return False


def validate_cli() -> bool:
    """Validate CLI functionality."""
    print("\n🔍 Validating CLI...")

    # Test that the CLI can be imported and run
    exit_code, stdout, stderr = run_command([
        sys.executable, "-c", "from jaf.cli import cli_main; print('CLI import OK')"
    ])

    if exit_code == 0:
        print("✅ CLI import: OK")
        return True
    else:
        print("❌ CLI import: FAILED")
        print(f"   stdout: {stdout}")
        print(f"   stderr: {stderr}")
        return False


def validate_examples() -> bool:
    """Validate that examples can be imported."""
    print("\n🔍 Validating examples...")

    root = Path(__file__).parent.parent.parent  # Go up to project root
    all_good = True

    examples = [
        "server_example.py",
        "iterative_search_agent.py"
    ]

    for example in examples:
        example_path = root / "examples" / example

        # Test syntax by compiling (don't execute server examples that might block)
        if "server" in example:
            # Just compile, don't execute server examples
            exit_code, stdout, stderr = run_command([
                sys.executable, "-c", f"compile(open('{example_path}').read(), '{example_path}', 'exec')"
            ])
        elif "iterative_search_agent" in example:
            # ADK example - just check syntax compilation, don't execute
            exit_code, stdout, stderr = run_command([
                sys.executable, "-c", f"compile(open('{example_path}').read(), '{example_path}', 'exec')"
            ])
        else:
            # Execute non-server examples
            exit_code, stdout, stderr = run_command([
                sys.executable, "-c", f"import sys; sys.path.insert(0, '{root / 'examples'}'); "
                f"exec(compile(open('{example_path}').read(), '{example_path}', 'exec'))"
            ])

        if exit_code == 0:
            print(f"✅ Example {example}: Syntax OK")
        else:
            print(f"❌ Example {example}: Syntax Error")
            print(f"   stderr: {stderr}")
            all_good = False

    return all_good


def run_quick_tests() -> bool:
    """Run a quick test to verify basic functionality."""
    print("\n🔍 Running quick functionality test...")

    try:
        # Test basic functionality
        test_code = '''
import asyncio
from jaf import Agent, RunState, RunConfig, run, generate_run_id, generate_trace_id
from jaf.core.types import Message
from jaf.providers.model import make_litellm_provider

# Create a simple agent
def test_instructions(state):
    return "You are a test assistant."

test_agent = Agent(
    name="test",
    instructions=test_instructions,
    tools=None
)

# Test that we can create the basic structures
run_id = generate_run_id()
trace_id = generate_trace_id()

state = RunState(
    run_id=run_id,
    trace_id=trace_id,
    messages=[Message(role='user', content='test')],
    current_agent_name="test",
    context={},
    turn_count=0
)

# Test model provider creation (without actually calling)
model_provider = make_litellm_provider("http://example.com", "test-key")

config = RunConfig(
    agent_registry={"test": test_agent},
    model_provider=model_provider,
    max_turns=1
)

print("Basic functionality test: PASSED")
'''

        exit_code, stdout, stderr = run_command([
            sys.executable, "-c", test_code
        ])

        if exit_code == 0 and "PASSED" in stdout:
            print("✅ Quick functionality test: PASSED")
            return True
        else:
            print("❌ Quick functionality test: FAILED")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return False

    except Exception as e:
        print(f"❌ Quick functionality test: ERROR - {e}")
        return False


def main():
    """Main validation function."""
    print("🚀 JAF Python Package Validation")
    print("=" * 50)

    all_checks = [
        ("Package Structure", validate_package_structure),
        ("Imports", validate_imports),
        ("Key Exports", validate_key_exports),
        ("Dependencies", validate_dependencies),
        ("CLI", validate_cli),
        ("Examples", validate_examples),
        ("Quick Tests", run_quick_tests),
    ]

    results = []

    for check_name, check_func in all_checks:
        print()
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name}: EXCEPTION - {e}")
            results.append((check_name, False))

    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = 0
    failed = 0

    for check_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name:20} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED! Package is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {failed} validation(s) failed. Please fix before publishing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
