#!/usr/bin/env python3
"""
Test server to validate the three critical fixes:
1. Turn count persistence across conversations
2. Execution time accuracy (not server uptime)
3. Metadata inclusion in tool responses
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaf.core.types import Agent, RunConfig, ToolSchema
from jaf.core.tools import function_tool
from jaf.core.tool_results import ToolResult, ToolResultStatus, ToolMetadata
from jaf.memory.factory import create_memory_provider_from_env
from jaf.server.main import run_server
from jaf.providers.model import make_litellm_provider
from pydantic import BaseModel


class TestToolArgs(BaseModel):
    """Arguments for the test tool."""

    message: str


@function_tool
async def test_tool(message: str, context) -> str:
    """A test tool that returns metadata for testing the fixes.

    Args:
        message: Test message to echo back
    """
    import json

    execution_start = time.time()

    # Simulate some work
    await asyncio.sleep(0.1)

    execution_time_ms = int((time.time() - execution_start) * 1000)

    result_data = {
        "echo": message,
        "timestamp": execution_start,
        "tool_name": "test_tool",
        "status": "success",
    }

    # Create metadata
    metadata = {
        "executionTimeMs": execution_time_ms,
        "toolName": "test_tool",
        "tool_version": "1.0.0",
        "test_metadata": "This validates metadata inclusion fix",
        "fix_validation": "metadata_inclusion",
        "execution_timestamp": execution_start,
    }

    # Return success format with metadata
    success_response = {"status": "success", "data": result_data, "metadata": metadata}

    return json.dumps(success_response, default=str)


async def main():
    """Main function to run the test server."""
    print("=== JAF Test Server for Fixes Validation ===")
    print("Testing: Turn count persistence, Execution time accuracy, Metadata inclusion")
    print()

    # Check environment variables
    print("Environment Check:")
    print(f"JAF_MEMORY_TYPE: {os.getenv('JAF_MEMORY_TYPE', 'NOT_SET')}")
    print(f"JAF_REDIS_URL: {os.getenv('JAF_REDIS_URL', 'NOT_SET')}")
    print(f"LITELLM_MODEL: {os.getenv('LITELLM_MODEL', 'NOT_SET')}")
    print()

    # Create memory provider from environment
    print("Creating Redis memory provider...")
    try:
        memory_result = await create_memory_provider_from_env()
        if hasattr(memory_result, "error"):
            print(f"✗ Failed to create memory provider: {memory_result.error}")
            return

        memory_provider = memory_result.data
        print("✓ Redis memory provider created successfully")
    except Exception as e:
        print(f"✗ Error creating memory provider: {e}")
        return

    # Create test agent with test tool
    test_agent = Agent(
        name="test_agent",
        instructions=lambda state: "You are a test agent for validating JAF fixes. When asked to test something, use the test_tool to demonstrate the fixes work correctly.",
        tools=[test_tool],
    )

    # Create model provider using environment variables
    litellm_url = os.getenv("LITELLM_URL", "http://localhost:4000")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "test-key")

    model_provider = make_litellm_provider(base_url=litellm_url, api_key=litellm_api_key)

    # Create run config
    run_config = RunConfig(
        agent_registry={"test_agent": test_agent},
        model_provider=model_provider,
        max_turns=5,
        model_override=os.getenv("LITELLM_MODEL", "gpt-4o"),
    )

    print("✓ Test agent created with test_tool")
    print("✓ Run configuration ready")
    print()

    # Start the server
    print("🚀 Starting JAF server on http://127.0.0.1:8000")
    print("📋 Available agent: test_agent")
    print("🧠 Memory provider: Redis")
    print("📖 API docs: http://127.0.0.1:8000/docs")
    print()
    print("Server is ready for testing the fixes!")
    print("Use curl commands to test:")
    print("1. Turn count persistence - make multiple requests with same conversation_id")
    print("2. Execution time accuracy - check execution_time_ms in responses")
    print("3. Metadata inclusion - use test_tool and check for metadata in responses")
    print()

    await run_server(
        agents={"test_agent": test_agent},
        run_config=run_config,
        host="127.0.0.1",
        port=8000,
        cors=True,
        default_memory_provider=memory_provider,
    )


if __name__ == "__main__":
    asyncio.run(main())
