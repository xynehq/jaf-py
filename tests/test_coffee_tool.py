import asyncio
import os
import sys
import socket
from typing import Any, Optional
from pydantic import BaseModel
import httpx
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, ".")

from jaf.core.tools import create_function_tool
from jaf.core.types import (
    Agent,
    Message,
    RunState,
    RunConfig,
    ModelConfig,
    generate_run_id,
    generate_trace_id,
)
from jaf.core.engine import run
from jaf.providers.model import make_litellm_provider


# Coffee API tool definition
class CoffeeArgs(BaseModel):
    coffee_type: Optional[str] = "hot"  # "hot" or "iced"


async def get_coffee_data(args: CoffeeArgs, context: Any) -> str:
    """Fetch coffee data from the sample API"""
    try:
        # Determine the API endpoint based on coffee type
        if args.coffee_type.lower() == "iced":
            url = "https://api.sampleapis.com/coffee/iced"
        else:
            url = "https://api.sampleapis.com/coffee/hot"

        print(f"[TEST DEBUG] Fetching coffee data from: {url}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            coffee_data = response.json()

        # Format the response nicely
        if isinstance(coffee_data, list) and len(coffee_data) > 0:
            # Show first 3 coffee items
            coffee_list = coffee_data[:3]
            formatted_coffees = []

            for coffee in coffee_list:
                name = coffee.get("title", "Unknown Coffee")
                description = coffee.get("description", "No description available")
                formatted_coffees.append(f"• {name}: {description}")

            result = f"COFFEE_API_RESULT: Found {len(coffee_data)} {args.coffee_type} coffee options. Here are the first 3:\n\n"
            result += "\n".join(formatted_coffees)

            if len(coffee_data) > 3:
                result += f"\n\n... and {len(coffee_data) - 3} more options available!"

            return result
        else:
            return f"COFFEE_API_RESULT: No {args.coffee_type} coffee data found"

    except httpx.TimeoutException:
        return "COFFEE_API_RESULT: Error - Coffee API request timed out"
    except httpx.HTTPStatusError as e:
        return f"COFFEE_API_RESULT: Error - Coffee API returned status {e.response.status_code}"
    except Exception as e:
        return f"COFFEE_API_RESULT: Error fetching coffee data: {str(e)}"


# Create coffee tool
coffee_tool = create_function_tool(
    {
        "name": "get_coffee_info",
        "description": 'Get information about coffee types from the API. Use "hot" for hot coffee or "iced" for iced coffee.',
        "execute": get_coffee_data,
        "parameters": CoffeeArgs,
    }
)


def coffee_test_instructions(state):
    return "You are a coffee assistant that can provide information about different coffee types using the coffee API."


def check_litellm_available():
    """Check if LiteLLM server is available."""
    try:
        litellm_url = os.getenv("LITELLM_URL", "http://0.0.0.0:4000/")
        # Parse URL to get host and port
        from urllib.parse import urlparse

        parsed = urlparse(litellm_url)
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 4000

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def check_env_available():
    """Check if required environment variables are available."""
    litellm_url = os.getenv("LITELLM_URL")
    # Either LITELLM_URL should be set, or we should have some API key
    return litellm_url is not None or os.getenv("LITELLM_API_KEY") is not None


skip_if_no_litellm = pytest.mark.skipif(
    not check_litellm_available() or not check_env_available(),
    reason="Skipping coffee tool test: LiteLLM server not available or environment variables (LITELLM_URL/LITELLM_API_KEY) not set. Please start LiteLLM server and configure .env file to run this test.",
)


@skip_if_no_litellm
async def test_coffee_tool_with_litellm():
    """Test that the coffee tool works with real LiteLLM."""

    print("☕ Testing coffee tool integration with real LiteLLM...")
    print("=" * 65)

    agent = Agent(
        name="CoffeeAgent",
        instructions=coffee_test_instructions,
        tools=[coffee_tool],
        model_config=ModelConfig(name="gemini-2.5-pro"),
    )

    print(f"✅ Agent created with tool: {agent.tools[0].schema.name}")

    # Use real LiteLLM provider
    litellm_url = os.getenv("LITELLM_URL", "http://0.0.0.0:4000/")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "")

    model_provider = make_litellm_provider(base_url=litellm_url, api_key=litellm_api_key)

    print(f"✅ Using real LiteLLM provider: {litellm_url}")

    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role="user", content="Show me some iced coffee options")],
        current_agent_name="CoffeeAgent",
        context={},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry={"CoffeeAgent": agent}, model_provider=model_provider, max_turns=3
    )

    print("🚀 Running JAF engine with coffee tool...")

    try:
        result = await run(initial_state, config)
        print(f"✅ Engine run completed successfully")

        print(f"DEBUG: Final Messages = {result.final_state.messages}")

        tool_messages = [m for m in result.final_state.messages if m.role == "tool"]
        assert len(tool_messages) > 0, "No tool messages found in the final state."
        assert "COFFEE_API_RESULT:" in tool_messages[0].content

        print(f"✅ Coffee tool execution successful!")
        print(f"   Tool result (first 200 chars): {tool_messages[0].content[:200]}...")

        # Check if we got real coffee data
        if "iced coffee options" in tool_messages[0].content.lower():
            print("✅ Real coffee API data retrieved successfully!")

    except Exception as e:
        print(f"❌ Engine run failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("\n" + "=" * 65)
    print("🎯 Coffee Tool + LiteLLM Integration Test Passed!")


if __name__ == "__main__":
    asyncio.run(test_coffee_tool_with_litellm())
