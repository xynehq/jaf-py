#!/usr/bin/env python3
"""
Global Test Runner for JAF Python

This script discovers and runs all test files scattered throughout the repository.
It provides a centralized way to execute all tests regardless of their location.

Usage:
    python run_all_tests.py [options]
    python run_all_tests.py --suite <suite_name>
    
Options:
    --suite         Run a predefined test suite (all, fast, unit, integration, a2a, memory, visualization, smoke, ci)
    --fast          Run tests without coverage reporting
    --verbose       Run with extra verbose output
    --markers       Filter tests by markers (e.g., unit, integration, a2a, memory, visualization)
    --path          Run tests only from specific path
    --list          List all discovered test files and exit
    --list-suites   List all available test suites and exit
    --parallel      Run tests in parallel (requires pytest-xdist)
    
Examples:
    python run_all_tests.py --suite fast
    python run_all_tests.py --suite a2a
    python run_all_tests.py --fast
    python run_all_tests.py --markers a2a
    python run_all_tests.py --path jaf/a2a/tests
    python run_all_tests.py --list
    python run_all_tests.py --list-suites
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Import test configuration
try:
    from test_config import TEST_SUITES, get_test_suite, list_test_suites
except ImportError:
    # Fallback if test_config is not available
    def get_test_suite(name: str):
        return None
    def list_test_suites():
        print("Test suites not available - test_config.py not found")
    TEST_SUITES = {}


def find_all_test_files(root_dir: Path = None) -> List[Path]:
    """Discover all test files in the repository."""
    if root_dir is None:
        root_dir = Path(__file__).parent

    test_patterns = ["test_*.py", "*_test.py"]
    test_files = []

    for pattern in test_patterns:
        test_files.extend(root_dir.rglob(pattern))

    # Filter out any files in virtual environments or build directories
    filtered_files = []
    excluded_dirs = {".venv", "venv", ".env", "env", "build", "dist", ".git", "__pycache__"}

    for test_file in test_files:
        if not any(excluded_dir in test_file.parts for excluded_dir in excluded_dirs):
            filtered_files.append(test_file)

    return sorted(filtered_files)


def get_test_directories() -> List[str]:
    """Get all directories that contain test files."""
    test_files = find_all_test_files()
    directories = set()

    for test_file in test_files:
        directories.add(str(test_file.parent))

    return sorted(directories)


def build_pytest_command(
    fast: bool = False,
    verbose: bool = False,
    markers: Optional[str] = None,
    path: Optional[str] = None,
    parallel: bool = False,
    extra_args: List[str] = None
) -> List[str]:
    """Build the pytest command with appropriate options."""
    cmd = ["python3", "-m", "pytest"]

    if path:
        cmd.append(path)

    if fast:
        # Fast mode: skip coverage and stop on first failure
        cmd.append("-x")
    else:
        # Normal mode: add coverage if pytest-cov is available
        try:
            import pytest_cov
            cmd.extend(["--cov=jaf", "--cov-report=term-missing"])
        except ImportError:
            pass  # Skip coverage if plugin not available

    if verbose:
        cmd.append("-vv")

    if markers:
        cmd.extend(["-m", markers])

    if parallel:
        # Only add parallel args if pytest-xdist is available
        try:
            import xdist
            cmd.extend(["-n", "auto"])
        except ImportError:
            print("⚠️ Warning: pytest-xdist not available, running tests sequentially")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def list_test_files():
    """List all discoverable test files."""
    print("🔍 Discovering test files across the repository...\n")

    test_files = find_all_test_files()

    if not test_files:
        print("❌ No test files found!")
        return

    print(f"📊 Found {len(test_files)} test files:\n")

    # Group by directory
    by_directory = {}
    for test_file in test_files:
        dir_name = str(test_file.parent)
        if dir_name not in by_directory:
            by_directory[dir_name] = []
        by_directory[dir_name].append(test_file.name)

    for directory, files in sorted(by_directory.items()):
        print(f"📁 {directory}/")
        for file in sorted(files):
            print(f"   └── {file}")
        print()


def run_tests(args: argparse.Namespace) -> int:
    """Run tests with the specified configuration."""
    if args.list:
        list_test_files()
        return 0

    if args.list_suites:
        list_test_suites()
        return 0

    # Handle test suite selection
    suite = None
    if args.suite:
        suite = get_test_suite(args.suite)
        if not suite:
            print(f"❌ Unknown test suite: {args.suite}")
            print(f"Available suites: {', '.join(TEST_SUITES.keys())}")
            return 1

    print("🧪 Running JAF Python Test Suite")
    print("=" * 50)

    if suite:
        print(f"📋 Test Suite: {suite.name}")
        print(f"📝 Description: {suite.description}")

        # Override args with suite configuration
        if suite.paths:
            args.path = " ".join(suite.paths)
        if suite.markers:
            args.markers = suite.markers
        if suite.fast:
            args.fast = True
        if suite.parallel:
            args.parallel = True
        if suite.extra_args:
            args.extra_args.extend(suite.extra_args)

    if args.path:
        print(f"📁 Running tests from: {args.path}")
    else:
        print("🔍 Running all discovered tests")

    if args.markers:
        print(f"🏷️  Filtered by markers: {args.markers}")

    print()

    # Build and execute pytest command
    cmd = build_pytest_command(
        fast=args.fast,
        verbose=args.verbose,
        markers=args.markers,
        path=args.path,
        parallel=args.parallel,
        extra_args=args.extra_args
    )

    print(f"🚀 Executing: {' '.join(cmd)}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Global test runner for JAF Python repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--suite",
        type=str,
        choices=list(TEST_SUITES.keys()) if TEST_SUITES else None,
        help="Run a predefined test suite"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run tests without coverage reporting for faster execution"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with extra verbose output"
    )

    parser.add_argument(
        "--markers",
        type=str,
        help="Filter tests by markers (unit, integration, a2a, memory, visualization)"
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Run tests only from specific path"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discovered test files and exit"
    )

    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List all available test suites and exit"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )

    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Validate path if provided
    if args.path and not Path(args.path).exists():
        print(f"❌ Error: Path '{args.path}' does not exist")
        return 1

    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
