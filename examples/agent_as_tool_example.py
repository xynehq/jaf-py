"""
Example demonstrating agent-as-tool functionality in JAF with LiteLLM.

This example shows how to use the agent.as_tool() method to create 
hierarchical agent orchestration patterns using LiteLLM proxy.

Prerequisites:
1. Set up LiteLLM proxy server with your preferred model
2. Create a .env file (see .env.example) with the following variables:
    - LITELLM_URL=<your_litellm_url>
    - LITELLM_API_KEY=<your_litellm_api_key>
    - LITELLM_MODEL=<your_model_name>

Usage:
CLI Demo Mode:
    # Load environment variables from .env (for development only)
    export $(cat .env | xargs)
    python examples/agent_as_tool_example.py

Server Mode (with curl examples):
    export $(cat .env | xargs)
    python examples/agent_as_tool_example.py --server

Security Notes:
- Never commit your actual .env file to version control. Ensure .env is listed in .gitignore.
- Rotate/revoke any previously exposed API keys immediately.
- This example will raise a clear error if required environment variables are missing.

The example creates:
- A Spanish translation agent using LiteLLM
- A French translation agent using LiteLLM  
- An orchestrator agent that uses the translation agents as tools
- Conditional tool enabling based on user preferences
- REST API endpoints for testing via curl
"""

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel

from jaf.core import (
    Agent,
    Message,
    ContentRole,
    RunConfig,
    RunState,
    ModelConfig,
    generate_run_id,
    generate_trace_id,
    run,
    create_json_output_extractor,
)
from jaf.providers.model import make_litellm_provider
from jaf import run_server, ConsoleTraceCollector


@dataclass(frozen=True)
class TranslationContext:
    """Context for translation operations."""
    user_id: str = "test_user"
    language_preference: str = "french_spanish"  # Can be "french_spanish" or "spanish_only"


class TranslationOutput(BaseModel):
    """Output format for translation agents."""
    translated_text: str
    source_language: str = "english"
    target_language: str


def create_litellm_provider():
    """Create a LiteLLM provider from environment variables."""
    litellm_url = os.environ.get("LITELLM_URL")
    litellm_api_key = os.environ.get("LITELLM_API_KEY")
    litellm_model = os.environ.get("LITELLM_MODEL")

    missing_vars = []
    if not litellm_url:
        missing_vars.append("LITELLM_URL")
    if not litellm_api_key:
        missing_vars.append("LITELLM_API_KEY")
    if not litellm_model:
        missing_vars.append("LITELLM_MODEL")
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please set them in your .env file.")

    print(f"📡 Using LiteLLM URL: {litellm_url}")
    print(f"🔑 API Key: {'Set' if litellm_api_key else 'Not set'}")
    print(f"🧠 Model: {litellm_model}")

    return make_litellm_provider(
        base_url=litellm_url,
        api_key=litellm_api_key,
        default_timeout=60.0
    )


def create_spanish_agent() -> Agent[TranslationContext, TranslationOutput]:
    """Create a Spanish translation agent."""
    model_name = os.environ.get("LITELLM_MODEL")
    
    return Agent(
        name="spanish_agent",
        instructions=lambda state: "You translate the user's message to Spanish. Always reply with a JSON object containing 'translated_text' and 'target_language' fields.",
        output_codec=TranslationOutput,
        model_config=ModelConfig(name=model_name, temperature=0.7)
    )


def create_french_agent() -> Agent[TranslationContext, TranslationOutput]:
    """Create a French translation agent."""
    model_name = os.environ.get("LITELLM_MODEL")
    
    return Agent(
        name="french_agent", 
        instructions=lambda state: "You translate the user's message to French. Always reply with a JSON object containing 'translated_text' and 'target_language' fields.",
        output_codec=TranslationOutput,
        model_config=ModelConfig(name=model_name, temperature=0.7)
    )


def french_enabled(context: TranslationContext, agent: Agent) -> bool:
    """Enable French translation only for french_spanish preference."""
    return context.language_preference == "french_spanish"


def create_orchestrator_agent() -> Agent[TranslationContext, str]:
    """Create an orchestrator agent that uses other agents as tools."""
    model_name = os.environ.get("LITELLM_MODEL")
    
    # Create specialized agents
    spanish_agent = create_spanish_agent()
    french_agent = create_french_agent()
    
    # Convert agents to tools
    spanish_tool = spanish_agent.as_tool(
        tool_name="translate_to_spanish",
        tool_description="Translate the user's message to Spanish",
        max_turns=3,
        custom_output_extractor=create_json_output_extractor(),
        is_enabled=True,  # Always enabled
        metadata={"language": "spanish", "category": "translation"}
    )
    
    french_tool = french_agent.as_tool(
        tool_name="translate_to_french",
        tool_description="Translate the user's message to French",
        max_turns=3,
        custom_output_extractor=create_json_output_extractor(),
        is_enabled=french_enabled,  # Conditionally enabled
        metadata={"language": "french", "category": "translation"}
    )
    
    # Create orchestrator with agent tools
    return Agent(
        name="orchestrator_agent",
        instructions=lambda state: (
            "You are a multilingual assistant. You use the tools given to you to respond to users. "
            "You must call ALL available tools to provide responses in different languages. "
            "You never respond in languages yourself, you always use the provided tools."
        ),
        tools=[spanish_tool, french_tool],
        model_config=ModelConfig(name=model_name, temperature=0.3)
    )


async def demonstrate_agent_as_tool():
    """Demonstrate the agent-as-tool functionality."""
    
    print("🤖 JAF Agent-as-Tool Example with LiteLLM")
    print("=" * 50)
    
    # Create LiteLLM provider
    try:
        model_provider = create_litellm_provider()
        print("✅ LiteLLM provider created successfully")
    except ValueError as e:
        print(f"❌ Error creating LiteLLM provider: {e}")
        print("Please check your LITELLM_URL environment variable")
        return
    
    # Create context
    context = TranslationContext(
        user_id="demo_user",
        language_preference="french_spanish"  # Enable both languages
    )
    
    # Create orchestrator agent with agent tools
    orchestrator = create_orchestrator_agent()
    
    # Create initial state
    initial_messages = [Message(
        role=ContentRole.USER,
        content="Hello, how are you?"
    )]
    
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=initial_messages,
        current_agent_name="orchestrator_agent",
        context=context,
        turn_count=0
    )
    
    # Create run configuration
    config = RunConfig(
        agent_registry={"orchestrator_agent": orchestrator},
        model_provider=model_provider,
        max_turns=10,
        default_tool_timeout=30.0
    )
    
    print(f"🌍 Testing with language preference: {context.language_preference}")
    print(f"📝 User input: {initial_messages[0].content}")
    print()
    
    # Run the orchestrator
    try:
        result = await run(initial_state, config)
        
        print("✅ Execution completed!")
        print(f"📊 Final turn count: {result.final_state.turn_count}")
        print(f"📤 Outcome status: {result.outcome.status}")
        
        if result.outcome.status == 'completed':
            if hasattr(result.outcome, 'output') and result.outcome.output:
                print(f"🎯 Final output: {result.outcome.output}")
            
            # Show the conversation flow
            print("\n💬 Conversation Flow:")
            for i, msg in enumerate(result.final_state.messages, 1):
                role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}.get(msg.role, "❓")
                # Safely handle non-string, None, or bytes content
                content = msg.content
                if content is None:
                    content_str = "None"
                elif isinstance(content, bytes):
                    content_str = content.decode("utf-8", errors="replace")
                elif not isinstance(content, str):
                    content_str = str(content)
                else:
                    content_str = content
                content_preview = content_str[:100] + "..." if len(content_str) > 100 else content_str
                print(f"  {i}. {role_emoji} {msg.role.upper()}: {content_preview}")
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"     🔧 Tool call: {tool_call.function.name}")
        else:
            print(f"❌ Error occurred: {result.outcome.error}")
            
    except Exception as e:
        print(f"💥 Exception during execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    
    # Test with different language preferences
    print("🔄 Testing with different language preference...")
    context_spanish_only = TranslationContext(
        user_id="demo_user", 
        language_preference="spanish_only"
    )
    
    state_spanish_only = replace(initial_state, context=context_spanish_only)
    
    print(f"🌍 Testing with language preference: {context_spanish_only.language_preference}")
    
    try:
        result2 = await run(state_spanish_only, config)
        print("✅ Second execution completed!")
        print(f"📊 Final turn count: {result2.final_state.turn_count}")
        
        # Count tool calls to see if French was disabled
        tool_messages = [msg for msg in result2.final_state.messages if msg.role == ContentRole.TOOL]
        print(f"🔧 Number of tool calls: {len(tool_messages)}")
        
    except Exception as e:
        print(f"💥 Exception during second execution: {e}")


async def start_server():
    """Start the JAF server with agent-as-tool functionality."""
    print("🚀 Starting JAF Agent-as-Tool Server with LiteLLM")
    print("=" * 60)
    
    # Create LiteLLM provider
    try:
        model_provider = create_litellm_provider()
        print("✅ LiteLLM provider created successfully")
    except ValueError as e:
        print(f"❌ Error creating LiteLLM provider: {e}")
        print("Please check your LITELLM_URL environment variable")
        return
    
    # Create orchestrator agent
    orchestrator = create_orchestrator_agent()
    
    # Set up tracing
    trace_collector = ConsoleTraceCollector()
    
    # Create run configuration
    run_config = RunConfig(
        agent_registry={"orchestrator_agent": orchestrator},
        model_provider=model_provider,
        max_turns=10,
        model_override=os.getenv('LITELLM_MODEL', 'qwen3-coder-480b'),
        on_event=trace_collector.collect,
        default_tool_timeout=30.0
    )
    
    # Server options
    port = int(os.getenv('PORT', '3001'))
    host = '127.0.0.1'
    
    print(f"\n📚 Try these example requests:")
    print("")
    print("1. Health Check:")
    print(f"   curl http://localhost:{port}/health")
    print("")
    print("2. List Agents:")
    print(f"   curl http://localhost:{port}/agents")
    print("")
    print("3. Translate with both Spanish and French (french_spanish preference):")
    print(f"   curl -X POST http://localhost:{port}/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"messages":[{"role":"user","content":"Hello, how are you today?"}],"agent_name":"orchestrator_agent","context":{"user_id":"demo","language_preference":"french_spanish"}}\'')
    print("")
    print("4. Translate with only Spanish (spanish_only preference):")
    print(f"   curl -X POST http://localhost:{port}/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"messages":[{"role":"user","content":"Good morning, how are you?"}],"agent_name":"orchestrator_agent","context":{"user_id":"demo","language_preference":"spanish_only"}}\'')
    print("")
    print("5. Translate a longer message:")
    print(f"   curl -X POST http://localhost:{port}/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"messages":[{"role":"user","content":"I hope you have a wonderful day filled with joy and happiness!"}],"agent_name":"orchestrator_agent","context":{"user_id":"demo","language_preference":"french_spanish"}}\'')
    print("")
    print("6. Chat using agent-specific endpoint:")
    print(f"   curl -X POST http://localhost:{port}/agents/orchestrator_agent/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"messages":[{"role":"user","content":"Thank you for your help"}],"context":{"user_id":"demo","language_preference":"french_spanish"}}\'')
    print("")
    print("🚀 Starting server...")
    
    try:
        # Start the server
        await run_server(
            agents=run_config.agent_registry,
            run_config=run_config,
            host=host,
            port=port,
            cors=False
        )
    except Exception as error:
        print(f"❌ Failed to start server: {error}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function - supports both CLI demo and server modes."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        await start_server()
    else:
        await demonstrate_agent_as_tool()


if __name__ == "__main__":
    asyncio.run(main())