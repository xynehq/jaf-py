"""
A2A Integration Example

This example demonstrates a complete A2A integration scenario where:
1. A server is started with multiple agents
2. A client connects and interacts with the agents
3. Both server and client run together in the same process

This is useful for testing and demonstrations.

Usage:
    python integration_example.py
"""

import asyncio
from typing import Any, Dict

from pydantic import BaseModel, Field

from jaf.a2a.agent import create_a2a_agent, create_a2a_tool
from jaf.a2a.client import (
    create_a2a_client,
    send_message_to_agent,
)
from jaf.a2a.server import create_a2a_server, create_server_config

# Import A2A functionality


# Mock model provider
class MockModelProvider:
    """Simple mock model provider for testing"""

    async def get_completion(self, state, agent, config):
        last_message = state.messages[-1].content if state.messages else ""
        return {
            "message": {
                "content": f"[{agent.name}] I received: '{last_message}'",
                "tool_calls": None,
            }
        }


# Tool definitions
class EchoArgs(BaseModel):
    message: str = Field(description="Message to echo back")


class CountArgs(BaseModel):
    text: str = Field(description="Text to count characters in")


async def echo_tool(args: EchoArgs, context) -> Dict[str, Any]:
    """Simple echo tool"""
    return {
        "result": f"Echo: {args.message}",
        "original": args.message,
        "length": len(args.message),
    }


async def count_tool(args: CountArgs, context) -> Dict[str, Any]:
    """Character counting tool"""
    char_count = len(args.text)
    word_count = len(args.text.split())

    return {
        "result": f"Text analysis: {char_count} characters, {word_count} words",
        "analysis": {"characters": char_count, "words": word_count, "text": args.text},
    }


def create_test_agents():
    """Create test agents for integration"""

    # Echo Agent
    echo_tool_obj = create_a2a_tool(
        "echo", "Echo back a message", EchoArgs.model_json_schema(), echo_tool
    )

    echo_agent = create_a2a_agent(
        "EchoBot",
        "An agent that echoes messages back",
        "You are an echo bot. Use the echo tool to repeat messages.",
        [echo_tool_obj],
    )

    # Counter Agent
    count_tool_obj = create_a2a_tool(
        "count_chars",
        "Count characters and words in text",
        CountArgs.model_json_schema(),
        count_tool,
    )

    counter_agent = create_a2a_agent(
        "CounterBot",
        "An agent that counts characters and words",
        "You are a text analysis bot. Use the counting tool to analyze text.",
        [count_tool_obj],
    )

    # Simple Chat Agent (no tools)
    chat_agent = create_a2a_agent(
        "ChatBot",
        "A simple conversational agent",
        "You are a friendly chat bot. Be helpful and conversational.",
        [],
    )

    return {"EchoBot": echo_agent, "CounterBot": counter_agent, "ChatBot": chat_agent}


async def start_test_server():
    """Start the test server"""
    print("🚀 Starting test A2A server...")

    agents = create_test_agents()

    # Create server configuration
    server_config = create_server_config(
        agents=agents,
        name="Test A2A Server",
        description="Integration test server with sample agents",
        host="localhost",
        port=3001,  # Use different port to avoid conflicts
    )

    server_config["model_provider"] = MockModelProvider()

    server = create_a2a_server(server_config)

    server_task = asyncio.create_task(server["start"]())

    # Give server time to start
    await asyncio.sleep(2)

    print("✅ Test server started on http://localhost:3001")

    return server, server_task


async def test_client_interactions(base_url: str):
    """Test various client interactions"""
    print(f"\n🔗 Testing client interactions with {base_url}")

    # Create client
    client = create_a2a_client(base_url)

    # Test cases for different agents
    test_cases = [
        {
            "agent": "EchoBot",
            "message": "Hello, echo bot!",
            "description": "Testing echo functionality",
        },
        {
            "agent": "CounterBot",
            "message": "Count the characters in this sentence please",
            "description": "Testing character counting",
        },
        {
            "agent": "ChatBot",
            "message": "Hi there, how are you doing today?",
            "description": "Testing general conversation",
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n📤 {test_case['description']}")
        print(f"   Agent: {test_case['agent']}")
        print(f"   Message: {test_case['message']}")

        try:
            response = await send_message_to_agent(client, test_case["agent"], test_case["message"])

            print(f"✅ Response: {response}")

            results.append(
                {
                    "agent": test_case["agent"],
                    "message": test_case["message"],
                    "response": response,
                    "success": True,
                }
            )

        except Exception as error:
            print(f"❌ Error: {error}")
            results.append(
                {
                    "agent": test_case["agent"],
                    "message": test_case["message"],
                    "error": str(error),
                    "success": False,
                }
            )

    return results


async def test_agent_discovery(base_url: str):
    """Test agent discovery"""
    print(f"\n🔍 Testing agent discovery at {base_url}")

    try:
        from jaf.a2a.client import discover_agents

        agent_card = await discover_agents(base_url)

        print(f"✅ Discovered server: {agent_card['name']}")
        print(f"📄 Description: {agent_card['description']}")

        skills = agent_card.get("skills", [])
        print(f"🛠️ Skills found: {len(skills)}")

        for skill in skills:
            print(f"   • {skill['name']}: {skill['description']}")

        return agent_card

    except Exception as error:
        print(f"❌ Discovery failed: {error}")
        return None


async def test_health_and_capabilities(base_url: str):
    """Test health checks and capabilities"""
    print(f"\n🏥 Testing health and capabilities at {base_url}")

    try:
        from jaf.a2a.client import check_a2a_health, create_a2a_client, get_a2a_capabilities

        client = create_a2a_client(base_url)

        # Health check
        health = await check_a2a_health(client)
        print(f"✅ Health status: {health.get('status', 'unknown')}")

        # Capabilities
        capabilities = await get_a2a_capabilities(client)
        methods = capabilities.get("supportedMethods", [])
        print(f"⚡ Supported methods: {', '.join(methods)}")

        return True

    except Exception as error:
        print(f"❌ Health/capabilities check failed: {error}")
        return False


async def main():
    """Main integration test"""
    print("🎯 A2A Integration Example")
    print("=" * 50)

    server_obj = None
    server_task = None

    try:
        # Start server
        server_obj, server_task = await start_test_server()

        base_url = "http://localhost:3001"

        # Test agent discovery
        agent_card = await test_agent_discovery(base_url)
        if not agent_card:
            print("❌ Agent discovery failed, stopping test")
            return

        # Test health and capabilities
        health_ok = await test_health_and_capabilities(base_url)
        if not health_ok:
            print("⚠️ Health checks failed, but continuing...")

        # Test client interactions
        results = await test_client_interactions(base_url)

        # Summary
        print("\n📊 Integration Test Summary")
        print("=" * 30)

        successful = sum(1 for r in results if r["success"])
        total = len(results)

        print(f"✅ Successful interactions: {successful}/{total}")

        if successful == total:
            print("🎉 All tests passed! A2A integration is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the errors above.")

        for result in results:
            status = "✅" if result["success"] else "❌"
            print(
                f"   {status} {result['agent']}: {result.get('response', result.get('error', 'Unknown'))[:60]}..."
            )

        print("\n🔧 Integration Features Tested:")
        print("   • Agent discovery via Agent Cards")
        print("   • Health checks and capabilities")
        print("   • Message sending to specific agents")
        print("   • Tool execution and responses")
        print("   • Error handling")

    except Exception as error:
        print(f"❌ Integration test failed: {error}")
        raise

    finally:
        # Clean up server
        if server_obj and server_task:
            print("\n🛑 Stopping test server...")
            try:
                server_task.cancel()
                await asyncio.sleep(1)
                print("✅ Server stopped")
            except Exception as cleanup_error:
                print(f"⚠️ Server cleanup error: {cleanup_error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Integration test stopped by user")
    except Exception as error:
        print(f"\n❌ Integration test failed: {error}")
        raise
