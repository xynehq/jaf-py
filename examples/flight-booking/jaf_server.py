#!/usr/bin/env python3
"""
JAF Server Integration for Flight Booking

This example shows how to integrate the flight booking agents with the JAF server,
exposing them via HTTP endpoints for web applications.
"""

import asyncio
import logging
import os
from typing import Dict, Any

from jaf.server.main import run_server
from jaf.core.tracing import ConsoleTraceCollector
from jaf.providers.model import make_litellm_provider

# Import agents from multi-agent example
from multi_agent import (
    coordinator_agent,
    search_specialist_agent,
    booking_specialist_agent,
    pricing_specialist_agent,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_server_config():
    """Create server configuration for flight booking system."""

    # Set up model provider (requires LiteLLM proxy running on localhost:4000)
    model_provider = make_litellm_provider(
        base_url=os.getenv("LITELLM_URL", "http://localhost:4000"),
        api_key=os.getenv("LITELLM_API_KEY"),
    )

    # Set up tracing
    trace_collector = ConsoleTraceCollector()

    # Agent registry with all flight booking agents
    agent_registry = {
        "FlightCoordinator": coordinator_agent,
        "SearchSpecialist": search_specialist_agent,
        "BookingSpecialist": booking_specialist_agent,
        "PricingSpecialist": pricing_specialist_agent,
    }

    # Import RunConfig from jaf.core.types
    from jaf.core.types import RunConfig

    # Create run config
    run_config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=10,
        on_event=trace_collector.collect,
    )

    return agent_registry, run_config


async def main():
    """Start the JAF server with flight booking agents."""
    print("🛫 Starting JAF Flight Booking Server")
    print("=" * 50)

    try:
        agent_registry, run_config = create_server_config()

        print("🌐 Server starting on http://127.0.0.1:3000")
        print("📋 Available endpoints:")
        print("  • GET  /health - Health check")
        print("  • GET  /agents - List available agents")
        print("  • POST /chat - Chat with default agent")
        print("  • POST /agents/{agent_name}/chat - Chat with specific agent")
        print("  • GET  /docs - API documentation")
        print("\n🤖 Available agents:")
        for agent_name in agent_registry.keys():
            print(f"  • {agent_name}")

        print("\n💡 Example curl commands:")
        print("""
# Chat with the flight coordinator
curl -X POST http://localhost:3000/agents/FlightCoordinator/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "I want to book a flight from LAX to JFK"}'

# Search for flights directly
curl -X POST http://localhost:3000/agents/SearchSpecialist/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Find flights from LAX to JFK departing tomorrow"}'

# Complete a booking
curl -X POST http://localhost:3000/agents/BookingSpecialist/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Book flight AA101 for John Doe"}'
        """)

        print("\n🚀 Starting server... (Press Ctrl+C to stop)")

        # Start the server with correct parameters
        await run_server(
            agents=agent_registry, run_config=run_config, host="127.0.0.1", port=3000, cors=True
        )

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        print(f"❌ Failed to start server: {e}")

        # Provide troubleshooting tips
        print("\n🔧 Troubleshooting:")
        print("1. Make sure LiteLLM proxy is running on localhost:4000")
        print("2. Check that port 3000 is available")
        print("3. Verify your API key configuration")
        print("4. Check the logs above for specific error details")


# Additional utility functions for server management
def validate_server_dependencies():
    """Validate that all server dependencies are available."""
    issues = []

    try:
        import litellm
    except ImportError:
        issues.append("❌ LiteLLM not installed - run: pip install litellm")

    try:
        import fastapi
    except ImportError:
        issues.append("❌ FastAPI not installed - run: pip install fastapi uvicorn")

    if issues:
        print("🚨 Dependency Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False

    print("✅ All dependencies satisfied")
    return True


def create_development_config():
    """Create a development-friendly server configuration."""

    # Mock model provider for development
    class MockModelProvider:
        async def get_completion(self, state, agent, config):
            return {
                "message": {
                    "content": f"This is a mock response from {agent.name}. In production, this would be powered by a real LLM.",
                    "tool_calls": None,
                }
            }

    # Simple trace collector for development
    class DevTraceCollector:
        def collect(self, event):
            print(f"🔍 Trace: {event.type}")

    agent_registry = {
        "FlightCoordinator": coordinator_agent,
        "SearchSpecialist": search_specialist_agent,
        "BookingSpecialist": booking_specialist_agent,
        "PricingSpecialist": pricing_specialist_agent,
    }

    # Import RunConfig from jaf.core.types
    from jaf.core.types import RunConfig

    # Create run config with mock provider
    run_config = RunConfig(
        agent_registry=agent_registry,
        model_provider=MockModelProvider(),
        max_turns=5,
        on_event=DevTraceCollector().collect,
    )

    return agent_registry, run_config


async def run_development_server():
    """Run the server in development mode with mock providers."""
    print("🔧 Starting JAF Flight Booking Server (Development Mode)")
    print("=" * 60)
    print("ℹ️  Using mock model provider - no external LLM required")

    agent_registry, run_config = create_development_config()

    print("🌐 Development server starting on http://127.0.0.1:3000")
    print("📋 This is a development version with mocked responses")
    print("🚀 Starting server... (Press Ctrl+C to stop)")

    try:
        await run_server(
            agents=agent_registry, run_config=run_config, host="127.0.0.1", port=3000, cors=True
        )
    except KeyboardInterrupt:
        print("\n👋 Development server stopped")


if __name__ == "__main__":
    import sys

    # Check command line arguments for development mode
    if len(sys.argv) > 1 and sys.argv[1] == "--dev":
        asyncio.run(run_development_server())
    else:
        # Validate dependencies first
        if validate_server_dependencies():
            asyncio.run(main())
        else:
            print("\n💡 To run in development mode (no external dependencies):")
            print("   python jaf_server.py --dev")
