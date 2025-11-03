"""
A2A Client Example

This example demonstrates how to connect to and interact with A2A-enabled agents
using the JAF A2A client library.

Usage:
    # Start the server first:
    python server_example.py

    # Then run this client:
    python client_example.py

The client will connect to http://localhost:3000 and demonstrate:
- Agent discovery via Agent Cards
- Sending messages to specific agents
- Streaming message responses
- Health checks and capabilities
- Error handling
"""

import asyncio
import json

# Import A2A client functionality
from jaf.a2a.client import (
    check_a2a_health,
    connect_to_a2a_agent,
    create_a2a_client,
    discover_agents,
    get_a2a_capabilities,
    send_message_to_agent,
    stream_message,
)


async def discover_server_agents(base_url: str):
    """Discover available agents on the server"""
    print("🔍 Discovering agents...")

    try:
        agent_card = await discover_agents(base_url)

        print(f"✅ Found server: {agent_card['name']}")
        print(f"📄 Description: {agent_card['description']}")
        print(f"🌐 URL: {agent_card['url']}")
        print(f"📡 Protocol: {agent_card['protocolVersion']}")

        skills = agent_card.get("skills", [])
        print(f"\n🛠️ Available skills ({len(skills)}):")
        for skill in skills:
            print(f"   • {skill['name']}: {skill['description']}")
            if skill.get("tags"):
                print(f"     Tags: {', '.join(skill['tags'])}")

        return agent_card

    except Exception as error:
        print(f"❌ Failed to discover agents: {error}")
        return None


async def check_server_health(client):
    """Check server health and capabilities"""
    print("\n🏥 Checking server health...")

    try:
        health = await check_a2a_health(client)
        print(f"✅ Server status: {health.get('status', 'unknown')}")
        print(f"📡 Protocol: {health.get('protocol', 'unknown')}")
        print(f"🔢 Version: {health.get('version', 'unknown')}")

        agents = health.get("agents", [])
        if agents:
            print(f"🤖 Available agents: {', '.join(agents)}")

        print("\n⚡ Getting capabilities...")
        capabilities = await get_a2a_capabilities(client)

        methods = capabilities.get("supportedMethods", [])
        if methods:
            print(f"📋 Supported methods: {', '.join(methods)}")

        transports = capabilities.get("supportedTransports", [])
        if transports:
            print(f"🚀 Supported transports: {', '.join(transports)}")

        return True

    except Exception as error:
        print(f"❌ Health check failed: {error}")
        return False


async def test_math_agent(client):
    """Test the MathTutor agent"""
    print("\n🧮 Testing MathTutor agent...")

    test_cases = [
        "What is 25 + 17?",
        "Calculate 144 / 12",
        "What is 2^8?",
        "Can you solve (15 + 5) * 3?",
    ]

    for question in test_cases:
        try:
            print(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "MathTutor", question)
            print(f"📥 Response: {response}")

        except Exception as error:
            print(f"❌ Error with question '{question}': {error}")


async def test_weather_agent(client):
    """Test the WeatherBot agent"""
    print("\n🌤️ Testing WeatherBot agent...")

    locations = [
        "What's the weather in London?",
        "How's the weather in Tokyo?",
        "Tell me about the weather in New York",
        "Weather forecast for Sydney please",
    ]

    for question in locations:
        try:
            print(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "WeatherBot", question)
            print(f"📥 Response: {response}")

        except Exception as error:
            print(f"❌ Error with question '{question}': {error}")


async def test_translator_agent(client):
    """Test the Translator agent"""
    print("\n🌍 Testing Translator agent...")

    translations = [
        "Translate 'Hello, how are you?' to Spanish",
        "Can you translate 'Good morning' to French?",
        "Translate 'Thank you very much' to German",
        "Convert 'Beautiful day' to Italian",
    ]

    for question in translations:
        try:
            print(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "Translator", question)
            print(f"📥 Response: {response}")

        except Exception as error:
            print(f"❌ Error with question '{question}': {error}")


async def test_streaming_response(client):
    """Test streaming message responses"""
    print("\n🌊 Testing streaming responses...")

    try:
        question = "Tell me about the weather in Paris and then calculate 50 * 24"
        print(f"📤 Streaming question: {question}")

        print("📥 Streaming response:")
        async for event in stream_message(client, question):
            print(f"   📡 Event: {json.dumps(event, indent=2)}")

    except Exception as error:
        print(f"❌ Streaming error: {error}")


async def test_general_assistant(client):
    """Test the general Assistant agent"""
    print("\n🤖 Testing general Assistant...")

    questions = [
        "Hello, what can you help me with?",
        "Tell me a joke",
        "What is the meaning of life?",
        "How can I be more productive?",
    ]

    for question in questions:
        try:
            print(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "Assistant", question)
            print(f"📥 Response: {response}")

        except Exception as error:
            print(f"❌ Error with question '{question}': {error}")


async def test_convenience_connection():
    """Test the convenience connection method"""
    print("\n🔗 Testing convenience connection...")

    try:
        # Use the convenience connection method
        connection = await connect_to_a2a_agent("http://localhost:3000")

        print("✅ Connected using connect_to_a2a_agent()")

        # Test the convenience methods
        response1 = await connection["ask"]("What is 10 + 15?")
        print(f"📥 Convenience ask: {response1}")

        # Test health check
        health = await connection["health"]()
        print(f"🏥 Health via convenience: {health.get('status', 'unknown')}")

        # Test capabilities
        capabilities = await connection["capabilities"]()
        methods = capabilities.get("supportedMethods", [])
        print(f"⚡ Methods via convenience: {len(methods)} methods")

        return True

    except Exception as error:
        print(f"❌ Convenience connection failed: {error}")
        return False


async def main():
    """Main function to run client examples"""
    base_url = "http://localhost:3000"

    print("🚀 Starting A2A Client Example...")
    print(f"🌐 Connecting to: {base_url}")

    # Discover agents first
    agent_card = await discover_server_agents(base_url)
    if not agent_card:
        print("❌ Cannot discover agents. Make sure the server is running.")
        return

    # Create client
    try:
        client = create_a2a_client(base_url)
        print(f"✅ Created A2A client for {base_url}")

    except Exception as error:
        print(f"❌ Failed to create client: {error}")
        return

    # Check server health
    if not await check_server_health(client):
        print("❌ Server health check failed. Continuing anyway...")

    # Test individual agents
    await test_math_agent(client)
    await test_weather_agent(client)
    await test_translator_agent(client)
    await test_general_assistant(client)

    # Test streaming
    await test_streaming_response(client)

    # Test convenience methods
    await test_convenience_connection()

    print("\n✅ Client example completed!")
    print("\n📊 Summary:")
    print("   • Successfully connected to A2A server")
    print("   • Discovered agent capabilities")
    print("   • Tested message sending to specific agents")
    print("   • Tested streaming responses")
    print("   • Validated health checks and capabilities")
    print("   • Used convenience connection methods")


if __name__ == "__main__":
    # Run the client example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Client example stopped by user")
    except Exception as error:
        print(f"\n❌ Client example failed: {error}")
        raise
