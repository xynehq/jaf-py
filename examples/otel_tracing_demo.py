import asyncio
import os
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from jaf import Agent, Message, ModelConfig, RunConfig, RunState, server
from jaf.core.engine import run
from jaf.core.types import ContentRole, generate_run_id, generate_trace_id
from jaf.core.tools import create_function_tool
from jaf.core.tracing import ConsoleTraceCollector, create_composite_trace_collector
from jaf.providers.model import make_litellm_provider

# Set the TRACE_COLLECTOR_URL to enable OpenTelemetry tracing
# Replace with your actual OTLP collector endpoint if needed
os.environ["TRACE_COLLECTOR_URL"] = "http://localhost:4318/v1/traces"


class Weather(BaseModel):
    """Get the weather for a location."""
    location: str = Field(..., description="The location to get the weather for.")
    unit: Annotated[
        Literal["celsius", "fahrenheit"],
        "The unit to use for the temperature.",
    ] = "celsius"


async def get_weather(args: Weather, context) -> str:
    """Get the weather for a location."""
    if "new york" in args.location.lower():
        return f"The weather in New York is 75°{args.unit}."
    return f"The weather in {args.location} is 25°{args.unit}."


async def main():
    """Run the agent and demonstrate OTEL tracing."""
    print("--- OpenTelemetry Tracing Demo ---")
    print(f"OTLP Exporter configured for: {os.getenv('TRACE_COLLECTOR_URL')}")
    print("Run an OTLP collector (like Jaeger) to view traces.")
    print("------------------------------------")

    # Create a composite collector that includes the console and the auto-configured OTEL collector
    trace_collector = create_composite_trace_collector(ConsoleTraceCollector())

    weather_tool = create_function_tool({
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "execute": get_weather,
        "parameters": Weather,
    })

    agent = Agent(
        name="weather_agent",
        instructions=lambda s: (
            "You are a function-calling AI model. You will be given a user's question and a set of tools. "
            "Your task is to follow these rules exactly: "
            "1. Examine the user's request. "
            "2. If you have a tool that can answer the request, call that tool. "
            "3. After the tool has been called and you have the result, your *final* action is to output the result to the user. "
            "Under no circumstances should you ever call the same tool more than once. "
            "Your response should be only the answer from the tool."
        ),
        tools=[weather_tool],
        model_config=ModelConfig(name="gemini-2.5-pro"),
    )

    # Get API key from environment or use a default.
    # Set the LITELLM_API_KEY environment variable to your actual key.
    litellm_api_key = os.getenv("LITELLM_API_KEY")
    if not litellm_api_key:
        print("\n---")
        print("Warning: LITELLM_API_KEY environment variable not set.")
        print("The current LiteLLM provider might require a valid API key.")
        print("---\n")


    config = RunConfig(
        agent_registry={"weather_agent": agent},
        model_provider=make_litellm_provider(
            base_url="",
            api_key=litellm_api_key
        ),
        on_event=trace_collector.collect,
    )

    # Create initial state
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content="what is the weather in new york?")],
        current_agent_name="weather_agent",
        context={},
        turn_count=0,
    )

    # Run the agent
    result = await run(initial_state, config)

    print("\n--- Agent Run Complete ---")
    if result.outcome.status == "completed":
        print(f"Final result: {result.outcome.output}")
    else:
        print(f"Run failed with error: {result.outcome.error}")
    print("--------------------------")


if __name__ == "__main__":
    asyncio.run(main())
