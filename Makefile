# JAF Python - Test Management Makefile
#
# This Makefile provides convenient commands to run tests across the entire repository.
# All test files scattered in different directories are automatically discovered.

.PHONY: help test test-all test-fast test-unit test-integration test-a2a test-memory test-viz test-ci test-smoke list-tests list-suites

# Default target
help:
	@echo "🧪 JAF Python Test Management"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make test          - Run all tests (default)"
	@echo "  make test-all      - Run all tests with full coverage"
	@echo "  make test-fast     - Run all tests quickly (no coverage)"
	@echo "  make test-unit     - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make test-a2a      - Run A2A protocol tests"
	@echo "  make test-memory   - Run memory provider tests"
	@echo "  make test-viz      - Run visualization tests"
	@echo "  make test-ci       - Run CI-suitable test suite"
	@echo "  make test-smoke    - Run smoke tests for quick verification"
	@echo ""
	@echo "  make list-tests    - List all discoverable test files"
	@echo "  make list-suites   - List all available test suites"
	@echo ""
	@echo "Examples:"
	@echo "  make test-fast     # Quick test run"
	@echo "  make test-a2a      # Test only A2A functionality"
	@echo "  make list-tests    # See all test files"

# Default test target - runs comprehensive test suite
test: test-all

# Run all tests with full coverage reporting
test-all:
	@echo "🧪 Running comprehensive test suite..."
	python run_all_tests.py --suite all

# Run tests quickly without coverage
test-fast:
	@echo "⚡ Running fast test suite..."
	python run_all_tests.py --suite fast

# Run only unit tests
test-unit:
	@echo "🔬 Running unit tests..."
	python run_all_tests.py --suite unit

# Run only integration tests
test-integration:
	@echo "🔗 Running integration tests..."
	python run_all_tests.py --suite integration

# Run A2A protocol tests
test-a2a:
	@echo "🤖 Running A2A protocol tests..."
	python run_all_tests.py --suite a2a

# Run memory provider tests
test-memory:
	@echo "💾 Running memory provider tests..."
	python run_all_tests.py --suite memory

# Run visualization tests
test-viz:
	@echo "📊 Running visualization tests..."
	python run_all_tests.py --suite visualization

# Run CI-suitable tests
test-ci:
	@echo "🚀 Running CI test suite..."
	python run_all_tests.py --suite ci

# Run smoke tests
test-smoke:
	@echo "💨 Running smoke tests..."
	python run_all_tests.py --suite smoke

# List all discoverable test files
list-tests:
	@echo "📋 Discovering all test files..."
	python run_all_tests.py --list

# List all available test suites
list-suites:
	@echo "📋 Available test suites..."
	python run_all_tests.py --list-suites

# Development helpers
install-test-deps:
	@echo "📦 Installing test dependencies..."
	pip install -e ".[dev]"

clean-test-cache:
	@echo "🧹 Cleaning test cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true

# Run specific test file (usage: make test-file FILE=tests/test_engine.py)
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Please specify FILE parameter. Example: make test-file FILE=tests/test_engine.py"; \
		exit 1; \
	fi
	@echo "🎯 Running specific test file: $(FILE)"
	python -m pytest $(FILE) -v

# Run tests with specific marker (usage: make test-marker MARKER=unit)
test-marker:
	@if [ -z "$(MARKER)" ]; then \
		echo "❌ Please specify MARKER parameter. Example: make test-marker MARKER=unit"; \
		exit 1; \
	fi
	@echo "🏷️ Running tests with marker: $(MARKER)"
	python run_all_tests.py --markers $(MARKER)