#!/usr/bin/env python3
"""
Simple JAF Handoff Example - Language Support System
Demonstrates handoffs between a triage agent and language specialist agents.
"""

import asyncio
import os
from typing import Any

# JAF imports
from jaf.core.types import Agent, RunState, RunConfig, Message, ContentRole, generate_run_id, generate_trace_id
from jaf.core.handoff import handoff_tool
from jaf.core.engine import run
from jaf.core.tracing import ConsoleTraceCollector
from jaf.providers.model import make_litellm_provider
from jaf.core.tools import create_function_tool

try:
    from pydantic import BaseModel
except ImportError:
    print("This example requires pydantic. Install with: pip install pydantic")
    exit(1)


# Simple tool for language agents
class TranslateArgs(BaseModel):
    text: str
    target_language: str

async def translate_text(args: TranslateArgs, context: Any) -> str:
    """Mock translation tool."""
    translations = {
        "french": {
            "hello": "bonjour",
            "goodbye": "au revoir",
            "thank you": "merci",
            "help": "aide",
            "how are you": "comment allez-vous"
        },
        "german": {
            "hello": "hallo",
            "goodbye": "auf wiedersehen",
            "thank you": "danke",
            "help": "hilfe",
            "how are you": "wie geht es dir"
        }
    }

    text_lower = args.text.lower()
    if args.target_language.lower() in translations:
        lang_dict = translations[args.target_language.lower()]
        translated = lang_dict.get(text_lower, f"[Translation of '{args.text}' to {args.target_language}]")
        return f"Translated '{args.text}' to {args.target_language}: '{translated}'"

    return f"Mock translation: '{args.text}' -> {args.target_language}"

# Create translation tool
translate_tool = create_function_tool({
    'name': 'translate',
    'description': 'Translate text to specified language',
    'execute': translate_text,
    'parameters': TranslateArgs
})


def create_language_support_agents():
    """Create simple language support agents for demonstration."""

    def triage_instructions(state: RunState) -> str:
        return """You are a language support triage agent.

Your job is to understand what language the customer needs help with and route them to the right specialist:

- If they mention "French", "français", or need French help → use handoff tool to transfer to "french_agent"
- If they mention "German", "deutsch", or need German help → use handoff tool to transfer to "german_agent"
- If they ask a simple question you can answer → respond directly

When using handoff tool:
- agent_name: "french_agent" or "german_agent"
- message: Brief summary of what the customer needs

Always be helpful and explain you're connecting them to the right language specialist."""

    def french_instructions(state: RunState) -> str:
        return """You are a French language specialist agent.

You help customers with:
- French translations
- French language questions
- French cultural information

You have tools:
- translate: Translate text to French

Be friendly and helpful. Speak some French when appropriate!
If the customer needs help with other languages, use handoff tool to route them appropriately."""

    def german_instructions(state: RunState) -> str:
        return """You are a German language specialist agent.

You help customers with:
- German translations
- German language questions
- German cultural information

You have tools:
- translate: Translate text to German

Be friendly and helpful. Speak some German when appropriate!
If the customer needs help with other languages, use handoff tool to route them appropriately."""

    # Create agents
    triage_agent = Agent(
        name="triage_agent",
        instructions=triage_instructions,
        tools=[handoff_tool],
        handoffs=["french_agent", "german_agent"]
    )

    french_agent = Agent(
        name="french_agent",
        instructions=french_instructions,
        tools=[translate_tool, handoff_tool],
        handoffs=["german_agent"]
    )

    german_agent = Agent(
        name="german_agent",
        instructions=german_instructions,
        tools=[translate_tool, handoff_tool],
        handoffs=["french_agent"]
    )

    return {
        "triage_agent": triage_agent,
        "french_agent": french_agent,
        "german_agent": german_agent
    }


async def demo_handoff(user_message: str):
    """Demonstrate handoff with a single message."""
    print(f"\nDemo: {user_message}")
    print("="*60)

    # Setup model provider
    model_provider = make_litellm_provider(
        base_url=os.getenv('LITELLM_BASE_URL'),
        api_key=os.getenv('LITELLM_API_KEY'),
        default_timeout=30.0
    )

    # Create agents
    agent_registry = create_language_support_agents()

    # Setup tracing
    trace_collector = ConsoleTraceCollector()

    # Create initial state
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content=user_message)],
        current_agent_name="triage_agent",
        context={"user_id": "demo_user"},
        turn_count=0
    )

    # Create run configuration
    config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=5,
        model_override=os.getenv('LITELLM_MODEL', 'gemini-2.5-pro'),
        on_event=trace_collector.collect
    )

    try:
        # Run the conversation
        result = await run(initial_state, config)

        print(f"\nResults:")
        print(f"Status: {result.outcome.status}")
        print(f"Final Agent: {result.final_state.current_agent_name}")
        print(f"Turn Count: {result.final_state.turn_count}")
        print(f"Final Output: {result.outcome.output}")

        # Show conversation flow
        print(f"\nConversation Flow:")
        print("-" * 40)
        for i, message in enumerate(result.final_state.messages):
            role_prefix = {"user": "[USER]", "assistant": "[AGENT]", "tool": "[TOOL]"}
            prefix = role_prefix.get(message.role, "[UNKNOWN]")
            content = str(message.content)[:100] + "..." if len(str(message.content)) > 100 else str(message.content)
            print(f"{i+1}. {prefix} {message.role}: {content}")

        return result.outcome.status == 'completed'

    except Exception as e:
        print(f"Error: {e}")
        return False


async def main():
    """Run the language support handoff demo."""
    print("JAF Language Support Handoff Demo")
    print("Simple example showing triage -> language specialist handoffs")
    print("="*60)

    # Demo scenarios
    scenarios = [
        "I need help translating 'hello' to French",
        "Can you help me with German translations?",
        "How do you say 'thank you' in French?",
        "I want to learn some German phrases",
        "What's the weather like today?"  # Should stay with triage
    ]

    print(f"\nRunning {len(scenarios)} demo scenarios...\n")

    results = []
    for scenario in scenarios:
        success = await demo_handoff(scenario)
        results.append(success)
        await asyncio.sleep(1)  # Brief pause between demos

    # Summary
    print(f"\nDemo Summary")
    print("="*30)
    print(f"Scenarios completed: {sum(results)}/{len(results)}")

    for i, (scenario, success) in enumerate(zip(scenarios, results)):
        status = "[PASS]" if success else "[FAIL]"
        print(f"{i+1}. {status} {scenario[:50]}...")

    if all(results):
        print(f"\nAll handoff demos completed successfully!")
        print("The handoff system is working perfectly!")
    else:
        print(f"\nSome demos had issues - check the logs above")


if __name__ == "__main__":
    print("Starting JAF Language Support Demo...")
    print("This demonstrates:")
    print("   - Triage agent routing requests")
    print("   - Handoffs to French/German specialists")
    print("   - Tool usage by specialist agents")
    print("   - Complete conversation flows")
    print(f"\nUsing model: {os.getenv('LITELLM_MODEL')}")
    print(f"Base URL: {os.getenv('LITELLM_BASE_URL')}")

    asyncio.run(main())