"""
JAF A2A Test Suite

Comprehensive tests for the Agent-to-Agent Communication Protocol implementation.

Test Structure:
- test_types.py: Core type validation and Pydantic model tests
- test_protocol.py: JSON-RPC protocol handler tests  
- test_client.py: HTTP client functionality tests
- test_agent.py: Agent creation and execution tests
- test_integration.py: End-to-end integration tests

Usage:
    # Run all A2A tests
    python -m pytest jaf/a2a/tests/
    
    # Run specific test file
    python -m pytest jaf/a2a/tests/test_types.py
    
    # Run with verbose output
    python -m pytest jaf/a2a/tests/ -v
    
    # Run with coverage
    python -m pytest jaf/a2a/tests/ --cov=jaf.a2a

Test Categories:
- Unit tests: Individual component functionality
- Integration tests: Component interaction testing
- Protocol tests: A2A protocol compliance
- Error handling: Exception and error scenarios
- Performance tests: Basic performance validation

Requirements:
- pytest
- pytest-asyncio (for async test support)
- pytest-cov (for coverage reporting, optional)
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all A2A tests with appropriate configuration"""
    test_dir = Path(__file__).parent

    # Configure pytest for async tests
    pytest_args = [
        str(test_dir),
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--asyncio-mode=auto",  # Auto async mode
        "-x",                   # Stop on first failure
    ]

    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend([
            "--cov=jaf.a2a",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    except ImportError:
        print("pytest-cov not available, skipping coverage report")

    # Run tests
    exit_code = pytest.main(pytest_args)
    return exit_code


def run_specific_test(test_name: str):
    """Run a specific test file or test function"""
    test_dir = Path(__file__).parent

    if not test_name.startswith("test_"):
        test_name = f"test_{test_name}"

    if not test_name.endswith(".py"):
        test_name = f"{test_name}.py"

    test_path = test_dir / test_name

    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return 1

    pytest_args = [
        str(test_path),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]

    return pytest.main(pytest_args)


def run_integration_tests_only():
    """Run only integration tests"""
    test_dir = Path(__file__).parent
    integration_test = test_dir / "test_integration.py"

    pytest_args = [
        str(integration_test),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]

    return pytest.main(pytest_args)


def run_unit_tests_only():
    """Run only unit tests (exclude integration)"""
    test_dir = Path(__file__).parent

    pytest_args = [
        str(test_dir),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--ignore=test_integration.py"
    ]

    return pytest.main(pytest_args)


# Test discovery for pytest
def pytest_configure(config):
    """Configure pytest for A2A tests"""
    # Add custom markers
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )


# Test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection for A2A tests"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


if __name__ == "__main__":
    """Run tests when module is executed directly"""
    import sys

    if len(sys.argv) > 1:
        test_arg = sys.argv[1]

        if test_arg == "integration":
            exit_code = run_integration_tests_only()
        elif test_arg == "unit":
            exit_code = run_unit_tests_only()
        else:
            exit_code = run_specific_test(test_arg)
    else:
        exit_code = run_all_tests()

    sys.exit(exit_code)
