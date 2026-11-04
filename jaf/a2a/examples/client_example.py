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
import logging

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

logger = logging.getLogger(__name__)


async def discover_server_agents(base_url: str):
    """Discover available agents on the server"""
    logger.info("🔍 Discovering agents...")

    try:
        agent_card = await discover_agents(base_url)

        logger.info(f"✅ Found server: {agent_card['name']}")
        logger.info(f"📄 Description: {agent_card['description']}")
        logger.info(f"🌐 URL: {agent_card['url']}")
        logger.info(f"📡 Protocol: {agent_card['protocolVersion']}")

        skills = agent_card.get("skills", [])
        logger.info(f"\n🛠️ Available skills ({len(skills)}):")
        for skill in skills:
            logger.info(f"   • {skill['name']}: {skill['description']}")
            if skill.get("tags"):
                logger.info(f"     Tags: {', '.join(skill['tags'])}")

        return agent_card

    except Exception as error:
        logger.error(f"❌ Failed to discover agents: {error}")
        return None


async def check_server_health(client):
    """Check server health and capabilities"""
    logger.info("\n🏥 Checking server health...")

    try:
        health = await check_a2a_health(client)
        logger.info(f"✅ Server status: {health.get('status', 'unknown')}")
        logger.info(f"📡 Protocol: {health.get('protocol', 'unknown')}")
        logger.info(f"🔢 Version: {health.get('version', 'unknown')}")

        agents = health.get("agents", [])
        if agents:
            logger.info(f"🤖 Available agents: {', '.join(agents)}")

        logger.info("\n⚡ Getting capabilities...")
        capabilities = await get_a2a_capabilities(client)

        methods = capabilities.get("supportedMethods", [])
        if methods:
            logger.info(f"📋 Supported methods: {', '.join(methods)}")

        transports = capabilities.get("supportedTransports", [])
        if transports:
            logger.info(f"🚀 Supported transports: {', '.join(transports)}")

        return True

    except Exception as error:
        logger.error(f"❌ Health check failed: {error}")
        return False


async def test_math_agent(client):
    """Test the MathTutor agent"""
    logger.info("\n🧮 Testing MathTutor agent...")

    test_cases = [
        "What is 25 + 17?",
        "Calculate 144 / 12",
        "What is 2^8?",
        "Can you solve (15 + 5) * 3?",
    ]

    for question in test_cases:
        try:
            logger.info(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "MathTutor", question)
            logger.info(f"📥 Response: {response}")

        except Exception as error:
            logger.error(f"❌ Error with question '{question}': {error}")


async def test_weather_agent(client):
    """Test the WeatherBot agent"""
    logger.info("\n🌤️ Testing WeatherBot agent...")

    locations = [
        "What's the weather in London?",
        "How's the weather in Tokyo?",
        "Tell me about the weather in New York",
        "Weather forecast for Sydney please",
    ]

    for question in locations:
        try:
            logger.info(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "WeatherBot", question)
            logger.info(f"📥 Response: {response}")

        except Exception as error:
            logger.error(f"❌ Error with question '{question}': {error}")


async def test_translator_agent(client):
    """Test the Translator agent"""
    logger.info("\n🌍 Testing Translator agent...")

    translations = [
        "Translate 'Hello, how are you?' to Spanish",
        "Can you translate 'Good morning' to French?",
        "Translate 'Thank you very much' to German",
        "Convert 'Beautiful day' to Italian",
    ]

    for question in translations:
        try:
            logger.info(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "Translator", question)
            logger.info(f"📥 Response: {response}")

        except Exception as error:
            logger.error(f"❌ Error with question '{question}': {error}")


async def test_streaming_response(client):
    """Test streaming message responses"""
    logger.info("\n🌊 Testing streaming responses...")

    try:
        question = "Tell me about the weather in Paris and then calculate 50 * 24"
        logger.info(f"📤 Streaming question: {question}")

        logger.info("📥 Streaming response:")
        async for event in stream_message(client, question):
            logger.info(f"   📡 Event: {json.dumps(event, indent=2)}")

    except Exception as error:
        logger.error(f"❌ Streaming error: {error}")


async def test_general_assistant(client):
    """Test the general Assistant agent"""
    logger.info("\n🤖 Testing general Assistant...")

    questions = [
        "Hello, what can you help me with?",
        "Tell me a joke",
        "What is the meaning of life?",
        "How can I be more productive?",
    ]

    for question in questions:
        try:
            logger.info(f"\n📤 Question: {question}")
            response = await send_message_to_agent(client, "Assistant", question)
            logger.info(f"📥 Response: {response}")

        except Exception as error:
            logger.error(f"❌ Error with question '{question}': {error}")


async def test_convenience_connection():
    """Test the convenience connection method"""
    logger.info("\n🔗 Testing convenience connection...")

    try:
        # Use the convenience connection method
        connection = await connect_to_a2a_agent("http://localhost:3000")

        logger.info("✅ Connected using connect_to_a2a_agent()")

        # Test the convenience methods
        response1 = await connection["ask"]("What is 10 + 15?")
        logger.info(f"📥 Convenience ask: {response1}")

        # Test health check
        health = await connection["health"]()
        logger.info(f"🏥 Health via convenience: {health.get('status', 'unknown')}")

        # Test capabilities
        capabilities = await connection["capabilities"]()
        methods = capabilities.get("supportedMethods", [])
        logger.info(f"⚡ Methods via convenience: {len(methods)} methods")

        return True

    except Exception as error:
        logger.error(f"❌ Convenience connection failed: {error}")
        return False


async def main():
    """Main function to run client examples"""
    base_url = "http://localhost:3000"

    logger.info("🚀 Starting A2A Client Example...")
    logger.info(f"🌐 Connecting to: {base_url}")

    # Discover agents first
    agent_card = await discover_server_agents(base_url)
    if not agent_card:
        logger.error("❌ Cannot discover agents. Make sure the server is running.")
        return

    # Create client
    try:
        client = create_a2a_client(base_url)
        logger.info(f"✅ Created A2A client for {base_url}")

    except Exception as error:
        logger.error(f"❌ Failed to create client: {error}")
        return

    # Check server health
    if not await check_server_health(client):
        logger.error("❌ Server health check failed. Continuing anyway...")

    # Test individual agents
    await test_math_agent(client)
    await test_weather_agent(client)
    await test_translator_agent(client)
    await test_general_assistant(client)

    # Test streaming
    await test_streaming_response(client)

    # Test convenience methods
    await test_convenience_connection()

    logger.info("\n✅ Client example completed!")
    logger.info("\n📊 Summary:")
    logger.info("   • Successfully connected to A2A server")
    logger.info("   • Discovered agent capabilities")
    logger.info("   • Tested message sending to specific agents")
    logger.info("   • Tested streaming responses")
    logger.info("   • Validated health checks and capabilities")
    logger.info("   • Used convenience connection methods")


if __name__ == "__main__":
    # Run the client example
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Client example stopped by user")
    except Exception as error:
        logger.error(f"\n❌ Client example failed: {error}")
        raise
