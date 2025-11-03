"""
JAF Parallel Language Agents Demo

This example demonstrates how to use multiple language-specific agents as tools that execute in parallel
when called simultaneously by an orchestrator agent. JAF automatically handles
parallel execution of multiple tool calls within a single turn.

This demo features German and French agents working in parallel.
"""

import asyncio
import os
import logging
from dataclasses import dataclass

# JAF Imports
from jaf import Agent, make_litellm_provider
from jaf.core.types import RunState, RunConfig, Message, generate_run_id, generate_trace_id
from jaf.core.engine import run

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("jaf_parallel_demo.log")],
)
logger = logging.getLogger(__name__)


def setup_litellm_provider():
    """Setup LiteLLM provider using environment variables."""
    base_url = os.getenv("LITELLM_BASE_URL", "https://grid.ai.juspay.net/")
    api_key = os.getenv("LITELLM_API_KEY")

    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is required")

    return make_litellm_provider(base_url, api_key)


print("JAF Parallel Language Agents Demo")
print("Using LiteLLM with Gemini 2.5 Pro")
print("German Agent & French Agent")
print("=" * 50)


# Create specialized language agents
def create_language_agents():
    """Create German and French language agents."""

    # German Agent
    german_agent = Agent(
        name="german_specialist",
        instructions=lambda state: """Du bist ein deutscher Sprachspezialist. 
        
Deine Aufgaben:
- Antworte IMMER auf Deutsch
- Übersetze gegebene Texte ins Deutsche
- Erkläre deutsche Kultur und Sprache
- Sei freundlich und hilfsbereit
- Verwende authentische deutsche Ausdrücke
        
Du hilfst Menschen dabei, deutsche Sprache und Kultur zu verstehen.""",
        tools=[],
    )

    # French Agent
    french_agent = Agent(
        name="french_specialist",
        instructions=lambda state: """Tu es un spécialiste de la langue française.
        
Tes tâches:
- Réponds TOUJOURS en français
- Traduis les textes donnés en français
- Explique la culture et la langue françaises
- Sois aimable et serviable
- Utilise des expressions françaises authentiques
        
Tu aides les gens à comprendre la langue et la culture françaises.""",
        tools=[],
    )

    return german_agent, french_agent


# Create the language agents
german_agent, french_agent = create_language_agents()

# Convert agents to tools using the as_tool() method
german_tool = german_agent.as_tool(
    tool_name="ask_german_specialist",
    tool_description="Ask the German language specialist to respond in German or translate to German",
)

french_tool = french_agent.as_tool(
    tool_name="ask_french_specialist",
    tool_description="Ask the French language specialist to respond in French or translate to French",
)


@dataclass
class LanguageContext:
    """Context for language operations."""

    user_id: str = "demo_user"
    request_id: str = "lang_demo_001"
    languages: list = None
    task_type: str = "translation"

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["german", "french"]


async def demo_parallel_execution():
    """Demonstrate parallel language agent execution."""

    # Setup model provider
    try:
        model_provider = setup_litellm_provider()
        print("LiteLLM provider setup successful")
        logger.info("LiteLLM provider configured successfully")
    except Exception as e:
        print(f"Failed to setup LiteLLM provider: {e}")
        logger.error(f"LiteLLM provider setup failed: {e}")
        return

    # Create orchestrator agent with parallel language tools
    orchestrator = Agent(
        name="language_orchestrator",
        instructions=lambda state: """You are a language orchestrator that coordinates multiple language specialists.

When given a message to translate or respond to:

1. Call BOTH language specialists in parallel in the same response
2. Use ask_german_specialist and ask_french_specialist simultaneously 
3. After receiving both responses, provide a summary comparing the responses
4. Be helpful and explain any cultural nuances between the languages

IMPORTANT: Always call both language tools in the same response to demonstrate parallel execution.""",
        tools=[german_tool, french_tool],
    )

    # Create context
    context = LanguageContext(
        user_id="demo_user",
        request_id="lang_demo_001",
        languages=["german", "french"],
        task_type="multilingual_response",
    )

    # Test message
    test_message = "Hello! How are you doing today? I hope you are having a wonderful time learning new languages!"

    print(f"\nRunning Parallel Language Agents Demo")
    print("-" * 40)
    print(f"Context: Languages={context.languages}, Task={context.task_type}")
    print(f"Message: {test_message}")
    logger.info(f"Starting demo with context: {context.__dict__}")
    logger.info(f"Test message: {test_message}")

    try:
        print("\nExecuting parallel language agents...")
        logger.info("Creating agent registry and starting parallel execution")

        # Create agent registry with all agents
        agent_registry = {
            "language_orchestrator": orchestrator,
            "german_specialist": german_agent,
            "french_specialist": french_agent,
        }
        logger.debug(f"Agent registry created with agents: {list(agent_registry.keys())}")

        # Create run state
        run_id = generate_run_id()
        trace_id = generate_trace_id()
        logger.info(f"Generated run_id: {run_id}, trace_id: {trace_id}")

        initial_state = RunState(
            run_id=run_id,
            trace_id=trace_id,
            messages=[
                Message(
                    role="user",
                    content=f"Please have both language specialists respond to this message in parallel: {test_message}",
                )
            ],
            current_agent_name="language_orchestrator",
            context=context.__dict__,
            turn_count=0,
        )
        logger.debug(f"Initial state created: {initial_state}")

        # Create run config
        config = RunConfig(
            agent_registry=agent_registry,
            model_provider=model_provider,
            max_turns=3,
            model_override="gemini-2.5-pro",
        )
        logger.info(f"Run config: max_turns={config.max_turns}, model={config.model_override}")

        print("Starting JAF execution...")
        print(
            "WATCH: JAF will automatically detect multiple tool calls and execute them in parallel"
        )
        logger.info("Calling JAF run() function - this will show parallel tool execution")

        # Start timing the execution
        import time

        start_time = time.time()

        result = await run(initial_state, config)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution completed in {execution_time:.2f} seconds")
        logger.info(
            f"JAF execution completed with status: {result.outcome.status} in {execution_time:.2f}s"
        )

        print("\nParallel Language Results:")
        print("=" * 60)
        logger.info(f"Final state contains {len(result.final_state.messages)} messages")

        # Show the conversation flow
        for i, message in enumerate(result.final_state.messages):
            logger.debug(
                f"Message {i}: role={message.role}, content_length={len(message.content) if message.content else 0}"
            )
            if message.role == "user":
                print(f"\nUser Request:")
                print(
                    message.content[:200] + "..." if len(message.content) > 200 else message.content
                )
            elif message.role == "assistant":
                print(f"\nAssistant Response:")
                print(message.content)
                print("-" * 40)

        print(f"\nFinal Status: {result.outcome.status}")
        logger.info(f"Demo completed successfully with {result.final_state.turn_count} turns")

        # Log detailed execution information
        if hasattr(result.final_state, "context"):
            logger.debug(f"Final context: {result.final_state.context}")

    except Exception as e:
        print(f"Error running parallel language demo: {e}")
        logger.error(f"Demo execution failed: {e}", exc_info=True)
        import traceback

        traceback.print_exc()

    print("\nDemo completed!")
    print("JAF handled parallel agent execution automatically via asyncio.gather()")
    print(f"Check 'jaf_parallel_demo.log' for detailed execution logs")


async def demo_key_concepts():
    """Demonstrate key parallel execution concepts."""

    print("\n\nKey Concepts Demonstrated:")
    print("=" * 50)

    print("Parallel Agent Execution:")
    print("   - Multiple language agents called simultaneously in one turn")
    print("   - JAF automatically uses asyncio.gather() for parallel execution")
    print("   - Language responses generated concurrently")

    print("\nAgent-as-Tool Pattern:")
    print("   - German and French agents converted to tools using .as_tool()")
    print("   - Orchestrator coordinates multiple language specialists")
    print("   - Enables multilingual responses in parallel")

    print("\nLanguage-Specific Context:")
    print("   - Each agent maintains its language and cultural context")
    print("   - Specialized instructions for authentic responses")
    print("   - Cultural nuances preserved in parallel execution")

    print("\nBenefits:")
    print("   - German responses: Authentic deutsche Sprache")
    print("   - French responses: Langue française authentique")
    print("   - Parallel processing reduces response time")
    print("   - Scalable for additional languages")


if __name__ == "__main__":
    print("This demonstrates JAF's parallel language agents using the as_tool() pattern\n")

    # Run the main demo
    asyncio.run(demo_parallel_execution())

    # Show key concepts
    asyncio.run(demo_key_concepts())
