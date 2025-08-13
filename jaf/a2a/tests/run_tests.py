#!/usr/bin/env python3
"""
A2A Test Runner

Comprehensive test runner for the JAF A2A implementation.
Provides various test execution modes and reporting options.

Usage:
    python run_tests.py [mode] [options]

Modes:
    all         - Run all tests (default)
    unit        - Run only unit tests
    integration - Run only integration tests
    types       - Run only type tests
    protocol    - Run only protocol tests
    client      - Run only client tests
    agent       - Run only agent tests

Options:
    --verbose   - Verbose output
    --coverage  - Generate coverage report
    --html      - Generate HTML coverage report
    --xml       - Generate XML coverage report for CI
    --quiet     - Minimal output
    --failfast  - Stop on first failure
    --parallel  - Run tests in parallel (if pytest-xdist available)

Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py unit --coverage    # Unit tests with coverage
    python run_tests.py integration -v     # Integration tests verbose
    python run_tests.py --parallel         # All tests in parallel
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List


def setup_environment():
    """Setup test environment and Python path"""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Set environment variables for testing
    os.environ["PYTHONPATH"] = str(project_root)
    os.environ["JAF_TEST_MODE"] = "1"


def check_dependencies():
    """Check if required test dependencies are available"""
    required_packages = ["pytest", "pytest-asyncio"]
    optional_packages = ["pytest-cov", "pytest-xdist", "pytest-html"]

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_required.append(package)

    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append(package)

    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"⚠️  Optional packages not available: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))

    return True


def build_pytest_args(mode: str, args: argparse.Namespace) -> List[str]:
    """Build pytest arguments based on mode and options"""
    test_dir = Path(__file__).parent
    pytest_args = []

    # Set test path based on mode
    if mode == "all":
        pytest_args.append(str(test_dir))
    elif mode == "unit":
        pytest_args.extend([
            str(test_dir),
            "--ignore=" + str(test_dir / "test_integration.py")
        ])
    elif mode == "integration":
        pytest_args.append(str(test_dir / "test_integration.py"))
    elif mode in ["types", "protocol", "client", "agent"]:
        pytest_args.append(str(test_dir / f"test_{mode}.py"))
    else:
        pytest_args.append(str(test_dir))

    # Add asyncio support
    pytest_args.extend([
        "--asyncio-mode=auto",
        "--tb=short"
    ])

    # Verbosity
    if args.verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")
    else:
        pytest_args.append("-v")  # Default to verbose for better feedback

    # Stop on failure
    if args.failfast:
        pytest_args.append("-x")

    # Parallel execution
    if args.parallel:
        try:
            import pytest_xdist
            pytest_args.extend(["-n", "auto"])
        except ImportError:
            print("⚠️  pytest-xdist not available, running sequentially")

    # Coverage options
    if args.coverage:
        try:
            import pytest_cov
            pytest_args.extend([
                "--cov=jaf.a2a",
                "--cov-report=term-missing"
            ])

            if args.html:
                pytest_args.append("--cov-report=html:htmlcov")

            if args.xml:
                pytest_args.append("--cov-report=xml")
        except ImportError:
            print("⚠️  pytest-cov not available, skipping coverage")

    # HTML report
    if args.html and not args.coverage:
        try:
            import pytest_html
            pytest_args.extend([
                "--html=test_report.html",
                "--self-contained-html"
            ])
        except ImportError:
            print("⚠️  pytest-html not available, skipping HTML report")

    return pytest_args


def run_tests(mode: str, args: argparse.Namespace) -> int:
    """Run tests with specified mode and arguments"""
    print(f"🚀 Running A2A tests in '{mode}' mode...")

    # Setup environment
    setup_environment()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Build pytest arguments
    pytest_args = build_pytest_args(mode, args)

    print(f"📋 Test command: pytest {' '.join(pytest_args)}")
    print("-" * 60)

    # Import and run pytest
    try:
        import pytest
        exit_code = pytest.main(pytest_args)
    except ImportError:
        print("❌ pytest not available")
        return 1
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1

    # Report results
    print("-" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")

    # Coverage report location
    if args.coverage and args.html:
        coverage_path = Path("htmlcov/index.html")
        if coverage_path.exists():
            print(f"📊 Coverage report: {coverage_path.absolute()}")

    return exit_code


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="JAF A2A Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "types", "protocol", "client", "agent"],
        help="Test mode to run (default: all)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML reports"
    )

    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report"
    )

    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )

    args = parser.parse_args()

    # Run tests
    exit_code = run_tests(args.mode, args)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
