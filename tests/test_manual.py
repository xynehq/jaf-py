"""
Test LiteLLM SDK provider with multiple providers: Vertex AI, OpenAI, and Gemini.
"""

import asyncio
import os
from dotenv import load_dotenv
from jaf.providers import make_litellm_sdk_provider
from jaf import Agent, RunConfig, RunState, Message
from jaf.core.types import generate_run_id, generate_trace_id, ContentRole

# Load environment variables
load_dotenv()


async def test_openai():
    print("🧪 Testing OpenAI with LiteLLM SDK Provider...")

    # Check if OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⏭️  Skipping OpenAI test - OPENAI_API_KEY not found in environment")
        return None

    # OpenAI provider
    openai_provider = make_litellm_sdk_provider(api_key=openai_key, model="gpt-3.5-turbo")

    print("✅ OpenAI provider created")

    # Create a test conversation
    state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[
            Message(
                role=ContentRole.USER,
                content="Explain the difference between AI and Machine Learning in one sentence.",
            )
        ],
        current_agent_name="OpenAIAgent",
        context={},
        turn_count=1,
    )

    agent = Agent(
        name="OpenAIAgent",
        instructions=lambda s: "You are a helpful AI assistant powered by OpenAI.",
        tools=[],
    )

    config = RunConfig(agent_registry={"OpenAIAgent": agent}, model_provider=openai_provider)

    # Test completion
    try:
        print("📤 Sending message to OpenAI...")
        result = await openai_provider.get_completion(state, agent, config)

        print("✅ OpenAI API call successful!")
        print(f"📥 Response: {result['message']['content']}")
        print(f"🔧 Model used: {result.get('model', 'unknown')}")

        if result.get("usage"):
            usage = result["usage"]
            print(
                f"📊 Token usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total"
            )

        return True

    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that OPENAI_API_KEY in .env file is valid")
        print("2. Ensure you have sufficient OpenAI credits")
        return False


async def test_gemini():
    print("\n🧪 Testing Google Gemini with LiteLLM SDK Provider...")

    # Check if Google API key is available
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("⏭️  Skipping Gemini test - GOOGLE_API_KEY not found in environment")
        return None

    # Gemini provider (consumer API)
    gemini_provider = make_litellm_sdk_provider(api_key=google_key, model="gemini/gemini-2.5-pro")

    print("✅ Gemini provider created")

    # Create a test conversation
    state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[
            Message(
                role=ContentRole.USER,
                content="What are the key advantages of transformer architecture in deep learning?",
            )
        ],
        current_agent_name="GeminiAgent",
        context={},
        turn_count=1,
    )

    agent = Agent(
        name="GeminiAgent",
        instructions=lambda s: "You are a helpful AI assistant powered by Google Gemini.",
        tools=[],
    )

    config = RunConfig(agent_registry={"GeminiAgent": agent}, model_provider=gemini_provider)

    # Test completion
    try:
        print("📤 Sending message to Google Gemini...")
        result = await gemini_provider.get_completion(state, agent, config)

        print("✅ Gemini API call successful!")
        print(f"📥 Response: {result['message']['content']}")
        print(f"🔧 Model used: {result.get('model', 'unknown')}")

        if result.get("usage"):
            usage = result["usage"]
            print(
                f"📊 Token usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total"
            )

        return True

    except Exception as e:
        print(f"❌ Gemini API call failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that GOOGLE_API_KEY in .env file is valid")
        print("2. Ensure you have access to Gemini API")
        print("3. Verify your Google Cloud project has Gemini API enabled")
        return False


async def test_vertex_ai():
    print("\n🧪 Testing Vertex AI with LiteLLM SDK Provider...")

    # Check if Vertex AI configuration is available
    vertex_project = os.getenv("VERTEX_PROJECT")
    vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not vertex_project:
        print("⏭️  Skipping Vertex AI test - VERTEX_PROJECT not found in environment")
        return None

    # Vertex AI provider
    vertex_provider = make_litellm_sdk_provider(
        model="vertex_ai/gemini-2.5-pro",
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

    print("✅ Vertex AI provider created")

    # Create a test conversation
    state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content="Hello! how is ML different from DL?")],
        current_agent_name="VertexAgent",
        context={},
        turn_count=1,
    )

    agent = Agent(
        name="VertexAgent",
        instructions=lambda s: "You are a helpful AI assistant powered by Google's Vertex AI.",
        tools=[],
    )

    config = RunConfig(agent_registry={"VertexAgent": agent}, model_provider=vertex_provider)

    # Test completion
    try:
        print("📤 Sending message to Vertex AI...")
        result = await vertex_provider.get_completion(state, agent, config)

        print("✅ Vertex AI API call successful!")
        print(f"📥 Response: {result['message']['content']}")
        print(f"🔧 Model used: {result.get('model', 'unknown')}")

        if result.get("usage"):
            usage = result["usage"]
            print(
                f"📊 Token usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total"
            )

        return True

    except Exception as e:
        print(f"❌ Vertex AI API call failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check VERTEX_PROJECT and VERTEX_LOCATION in .env file")
        print("2. Authenticate with: gcloud auth application-default login")
        print("3. Ensure your Google Cloud Project has Vertex AI enabled")
        print("4. Verify you have access to the specified model in your region")
        return False


async def test_azure_openai():
    print("\n🧪 Testing Azure OpenAI with LiteLLM SDK Provider...")

    # Check if Azure configuration is available
    azure_api_key = os.getenv("AZURE_API_KEY")
    azure_api_base = os.getenv("AZURE_API_BASE")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT")
    azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

    if not azure_api_key or not azure_api_base or not azure_deployment:
        print("⏭️  Skipping Azure OpenAI test - Missing required environment variables")
        print("     Required: AZURE_API_KEY, AZURE_API_BASE, AZURE_DEPLOYMENT")
        return None

    # Azure OpenAI provider using LiteLLM SDK
    azure_provider = make_litellm_sdk_provider(
        model=f"azure/{azure_deployment}",  # Use azure/deployment-name format
        api_key=azure_api_key,
        api_base=azure_api_base,
        api_version=azure_api_version,
    )

    print("✅ Azure OpenAI provider created")

    # Create a test conversation
    state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[
            Message(
                role=ContentRole.USER, content="Explain the concept of neural networks in detail."
            )
        ],
        current_agent_name="AzureAgent",
        context={},
        turn_count=1,
    )

    agent = Agent(
        name="AzureAgent",
        instructions=lambda s: "You are a helpful AI assistant powered by Azure OpenAI.",
        tools=[],
    )

    config = RunConfig(agent_registry={"AzureAgent": agent}, model_provider=azure_provider)

    # Test completion
    try:
        print("📤 Sending message to Azure OpenAI...")
        result = await azure_provider.get_completion(state, agent, config)

        print("✅ Azure OpenAI API call successful!")
        print(f"📥 Response: {result['message']['content']}")
        print(f"🔧 Model used: {result.get('model', 'unknown')}")

        if result.get("usage"):
            usage = result["usage"]
            print(
                f"📊 Token usage: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)} total"
            )

        return True

    except Exception as e:
        print(f"❌ Azure OpenAI API call failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that AZURE_API_KEY in .env file is valid")
        print("2. Verify AZURE_API_BASE is correct (e.g., https://your-resource.openai.azure.com)")
        print("3. Ensure AZURE_DEPLOYMENT matches your deployment name")
        print("4. Check AZURE_API_VERSION is supported")
        print("5. Verify your Azure OpenAI resource has the deployment enabled")
        return False


async def run_all_tests():
    print("🚀 Testing LiteLLM SDK Provider with Multiple AI Providers")
    print("=" * 60)

    # Test all providers
    results = []

    # print("\n1️⃣ OpenAI Test:")
    # results.append(await test_openai())

    # print("\n2️⃣ Google Gemini Test:")
    # results.append(await test_gemini())

    # print("\n3️⃣ Vertex AI Test:")
    # results.append(await test_vertex_ai())

    print("\n4️⃣ Azure OpenAI Test:")
    results.append(await test_azure_openai())

    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary:")
    providers = ["OpenAI", "Google Gemini", "Vertex AI", "Azure OpenAI"]

    tested_count = 0
    passed_count = 0

    for i, (provider, result) in enumerate(zip(providers, results)):
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
            tested_count += 1
            passed_count += 1
        else:
            status = "❌ FAILED"
            tested_count += 1
        print(f"{i + 1}. {provider}: {status}")

    if tested_count == 0:
        print("\n⚠️  No providers tested - check your .env file configuration")
    else:
        print(f"\n🏆 Results: {passed_count}/{tested_count} tested providers working")

        if passed_count == tested_count:
            print("🎉 All tested providers working! LiteLLM SDK integration is successful!")
        else:
            print("💡 Check API keys and authentication for failed providers")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
