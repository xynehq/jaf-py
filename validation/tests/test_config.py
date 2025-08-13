"""
Test Configuration and Predefined Test Suites

This module provides predefined test configurations for different scenarios.
"""

from typing import Dict, List, Optional


class TestSuite:
    """Configuration for a test suite."""

    def __init__(
        self,
        name: str,
        description: str,
        paths: List[str] = None,
        markers: Optional[str] = None,
        fast: bool = False,
        parallel: bool = False,
        extra_args: List[str] = None
    ):
        self.name = name
        self.description = description
        self.paths = paths or []
        self.markers = markers
        self.fast = fast
        self.parallel = parallel
        self.extra_args = extra_args or []


# Predefined test suites
TEST_SUITES: Dict[str, TestSuite] = {
    "all": TestSuite(
        name="all",
        description="Run all tests in the repository",
        fast=False,
        parallel=False
    ),

    "fast": TestSuite(
        name="fast",
        description="Run all tests quickly without coverage",
        fast=True,
        parallel=True
    ),

    "unit": TestSuite(
        name="unit",
        description="Run only unit tests",
        markers="unit",
        fast=True
    ),

    "integration": TestSuite(
        name="integration",
        description="Run only integration tests",
        markers="integration"
    ),

    "core": TestSuite(
        name="core",
        description="Run core JAF framework tests",
        paths=["tests/test_engine.py", "tests/test_validation.py"]
    ),

    "a2a": TestSuite(
        name="a2a",
        description="Run A2A protocol tests",
        paths=["jaf/a2a/tests"],
        markers="a2a"
    ),

    "memory": TestSuite(
        name="memory",
        description="Run memory provider tests",
        paths=["jaf/a2a/memory/tests"],
        markers="memory"
    ),

    "visualization": TestSuite(
        name="visualization",
        description="Run visualization tests",
        paths=["tests/visualization"],
        markers="visualization"
    ),

    "smoke": TestSuite(
        name="smoke",
        description="Run basic smoke tests to verify system health",
        fast=True,
        extra_args=["--maxfail=1", "-q"]
    ),

    "ci": TestSuite(
        name="ci",
        description="Run tests suitable for CI/CD pipeline",
        fast=False,
        parallel=True,
        extra_args=["--tb=short", "--maxfail=5"]
    )
}


def get_test_suite(name: str) -> Optional[TestSuite]:
    """Get a predefined test suite by name."""
    return TEST_SUITES.get(name)


def list_test_suites() -> None:
    """Print all available test suites."""
    print("📋 Available Test Suites:")
    print("=" * 50)

    for name, suite in TEST_SUITES.items():
        print(f"\n🧪 {name}")
        print(f"   {suite.description}")

        if suite.paths:
            print(f"   📁 Paths: {', '.join(suite.paths)}")

        if suite.markers:
            print(f"   🏷️  Markers: {suite.markers}")

        features = []
        if suite.fast:
            features.append("fast")
        if suite.parallel:
            features.append("parallel")
        if suite.extra_args:
            features.append(f"extra args: {' '.join(suite.extra_args)}")

        if features:
            print(f"   ⚡ Features: {', '.join(features)}")


if __name__ == "__main__":
    list_test_suites()
