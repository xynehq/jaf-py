#!/usr/bin/env python3
"""
Multi-Agent Flight Booking System

This example demonstrates functional composition and multi-agent coordination
for complex flight booking scenarios using the new JAF API.
"""

import asyncio
import json
import os
from typing import Any, Dict

from pydantic import BaseModel, Field

from jaf.core.types import ContentRole
from jaf import (
    Agent,
    Message,
    ModelConfig,
    RunConfig,
    RunState,
    ToolSource,
    create_function_tool,
    generate_run_id,
    generate_trace_id,
    run,
)
from jaf.core.tool_results import ToolResponse, ToolResult

# Import tools from the main index file
from index import (
    search_flights_tool,
    check_seat_availability_tool,
    book_flight_tool,
    check_flight_status_tool,
    cancel_booking_tool,
    FlightSearchArgs,
    BookFlightArgs,
)


# Handoff tool for agent coordination
class HandoffArgs(BaseModel):
    """Arguments for agent handoff."""

    target_agent: str = Field(description="Target agent to hand off to")
    context: str = Field(description="Context information for the target agent")
    reason: str = Field(description="Reason for handoff")


async def handoff_execute(args: HandoffArgs, context: Any) -> ToolResult:
    """Execute handoff to another agent."""
    return ToolResponse.success(
        {
            "handoff_to": args.target_agent,
            "context": args.context,
            "reason": args.reason,
            "message": f"Handing off to {args.target_agent}: {args.reason}",
        }
    )


handoff_tool = create_function_tool(
    {
        "name": "handoff",
        "description": "Hand off conversation to a specialized agent",
        "execute": handoff_execute,
        "parameters": HandoffArgs,
        "metadata": {"category": "coordination", "priority": "high"},
        "source": ToolSource.NATIVE,
    }
)


# Specialized Agent 1: Flight Search Specialist
def search_specialist_instructions(state: RunState) -> str:
    """Instructions for the flight search specialist."""
    return """You are a flight search specialist. Your job is to:

1. Help users find the best flights based on their criteria
2. Provide detailed flight information including prices, times, and availability
3. Compare different options and make recommendations
4. Hand off to the booking specialist when user is ready to book
5. Hand off to the pricing specialist for complex pricing questions

When users want to proceed with booking, use the handoff tool to transfer them to the BookingSpecialist.
When users have complex pricing or fare questions, hand off to the PricingSpecialist."""


search_specialist_agent = Agent(
    name="SearchSpecialist",
    instructions=search_specialist_instructions,
    tools=[
        search_flights_tool,
        check_seat_availability_tool,
        check_flight_status_tool,
        handoff_tool,
    ],
    handoffs=["BookingSpecialist", "PricingSpecialist"],
    model_config=ModelConfig(name=os.getenv("LITELLM_MODEL", "gemini-2.0-flash"), temperature=0.2),
)


# Specialized Agent 2: Booking Specialist
def booking_specialist_instructions(state: RunState) -> str:
    """Instructions for the booking specialist."""
    return """You are a flight booking specialist. Your job is to:

1. Complete flight bookings for customers
2. Handle booking confirmations and provide booking details
3. Manage booking cancellations and changes
4. Provide booking-related customer service
5. Hand off back to search specialist if users need to find different flights

Always confirm booking details with the customer before proceeding.
If customers need to search for different flights, hand off to the SearchSpecialist."""


booking_specialist_agent = Agent(
    name="BookingSpecialist",
    instructions=booking_specialist_instructions,
    tools=[book_flight_tool, cancel_booking_tool, check_flight_status_tool, handoff_tool],
    handoffs=["SearchSpecialist", "PricingSpecialist"],
    model_config=ModelConfig(name=os.getenv("LITELLM_MODEL", "gemini-2.0-flash"), temperature=0.1),
)


# Specialized Agent 3: Pricing Specialist
def pricing_specialist_instructions(state: RunState) -> str:
    """Instructions for the pricing specialist."""
    return """You are a flight pricing and fare specialist. Your job is to:

1. Explain flight pricing, fare rules, and restrictions
2. Help customers understand different fare classes
3. Provide information about baggage fees, change fees, and cancellation policies
4. Compare prices across different flights and airlines
5. Hand off to booking specialist when customer is ready to book
6. Hand off to search specialist if customer needs different flight options

You have deep knowledge of airline pricing strategies and can help customers make informed decisions."""


pricing_specialist_agent = Agent(
    name="PricingSpecialist",
    instructions=pricing_specialist_instructions,
    tools=[search_flights_tool, check_seat_availability_tool, handoff_tool],
    handoffs=["SearchSpecialist", "BookingSpecialist"],
    model_config=ModelConfig(name=os.getenv("LITELLM_MODEL", "gemini-2.0-flash"), temperature=0.3),
)


# Coordinator Agent - Entry point that routes to specialists
def coordinator_instructions(state: RunState) -> str:
    """Instructions for the coordinator agent."""
    return """You are the main flight booking coordinator. You help customers by:

1. Understanding their travel needs and requirements
2. Routing them to the appropriate specialist:
   - SearchSpecialist: For finding and comparing flights
   - BookingSpecialist: For completing bookings and managing reservations
   - PricingSpecialist: For fare questions and pricing information

Analyze the customer's request and hand off to the most appropriate specialist.
Always explain why you're transferring them to help set expectations."""


coordinator_agent = Agent(
    name="Coordinator",
    instructions=coordinator_instructions,
    tools=[handoff_tool],
    handoffs=["SearchSpecialist", "BookingSpecialist", "PricingSpecialist"],
    model_config=ModelConfig(name=os.getenv("LITELLM_MODEL", "gemini-2.0-flash"), temperature=0.2),
)


# Higher-order functions for functional composition
def with_logging(tool_func):
    """Higher-order function that adds logging to tool execution."""

    async def logged_execute(args, context):
        print(f"🔧 Executing tool: {tool_func.__name__}")
        print(f"📋 Args: {args}")
        result = await tool_func(args, context)
        print(f"✅ Result: {result.status}")
        return result

    return logged_execute


def with_retry(tool_func, max_retries=3):
    """Higher-order function that adds retry logic to tool execution."""

    async def retry_execute(args, context):
        for attempt in range(max_retries):
            try:
                result = await tool_func(args, context)
                if result.status == "success":
                    return result
                if attempt == max_retries - 1:
                    return result
                print(f"⚠️ Attempt {attempt + 1} failed, retrying...")
            except Exception as e:
                if attempt == max_retries - 1:
                    return ToolResponse.error(f"Failed after {max_retries} attempts: {str(e)}")
                print(f"⚠️ Attempt {attempt + 1} failed with exception, retrying...")
        return result

    return retry_execute


def with_cache(tool_func):
    """Higher-order function that adds caching to tool execution."""
    cache = {}

    async def cached_execute(args, context):
        # Create cache key from args
        cache_key = str(args) if hasattr(args, "__dict__") else str(args)

        if cache_key in cache:
            print(f"💾 Cache hit for {tool_func.__name__}")
            return cache[cache_key]

        result = await tool_func(args, context)
        if result.status == "success":
            cache[cache_key] = result
            print(f"💾 Cached result for {tool_func.__name__}")

        return result

    return cached_execute


# Composed validator functions
def compose_validators(*validators):
    """Compose multiple validation functions into one."""

    def composed_validator(data):
        for validator in validators:
            result = validator(data)
            if not result.get("is_valid", False):
                return result
        return {"is_valid": True}

    return composed_validator


def validate_airport_code(data):
    """Validate airport code format."""
    if not hasattr(data, "origin") or not hasattr(data, "destination"):
        return {"is_valid": False, "error": "Missing airport codes"}

    for code in [data.origin, data.destination]:
        if len(code) != 3 or not code.isalpha():
            return {"is_valid": False, "error": f"Invalid airport code: {code}"}

    return {"is_valid": True}


def validate_passenger_count(data):
    """Validate passenger count."""
    if hasattr(data, "passengers") and data.passengers <= 0:
        return {"is_valid": False, "error": "Passenger count must be positive"}
    return {"is_valid": True}


# Example of functional composition in action
async def main():
    """Demonstrate multi-agent coordination with functional composition."""
    print("🎯 Multi-Agent Flight Booking System Demo")
    print("=" * 60)

    # Create mock model provider
    class MockModelProvider:
        def __init__(self):
            self.responses = {
                "Coordinator": "I'll help you find the perfect flight! Let me connect you with our flight search specialist who can help you explore available options.",
                "SearchSpecialist": "I found several great flight options for you! Would you like me to transfer you to our booking specialist to complete your reservation?",
                "BookingSpecialist": "Perfect! I can help you complete that booking. Let me confirm the details and process your reservation.",
                "PricingSpecialist": "I can explain all the fare options and help you understand the pricing structure for these flights.",
            }

        async def get_completion(self, state, agent, config):
            agent_name = agent.name
            content = self.responses.get(agent_name, "I'm here to help!")

            # Simulate handoff for demonstration
            tool_calls = None
            if agent_name == "Coordinator":
                tool_calls = [
                    {
                        "id": "handoff1",
                        "type": "function",
                        "function": {
                            "name": "handoff",
                            "arguments": json.dumps(
                                {
                                    "target_agent": "SearchSpecialist",
                                    "context": "Customer wants to find flights",
                                    "reason": "Route to search specialist for flight options",
                                }
                            ),
                        },
                    }
                ]
            elif agent_name == "SearchSpecialist" and state.turn_count > 0:
                tool_calls = [
                    {
                        "id": "handoff2",
                        "type": "function",
                        "function": {
                            "name": "handoff",
                            "arguments": json.dumps(
                                {
                                    "target_agent": "BookingSpecialist",
                                    "context": "Customer ready to book flight AA101",
                                    "reason": "Ready to complete booking",
                                }
                            ),
                        },
                    }
                ]

            return {"message": {"content": content, "tool_calls": tool_calls}}

    # Set up multi-agent configuration
    agent_registry = {
        "Coordinator": coordinator_agent,
        "SearchSpecialist": search_specialist_agent,
        "BookingSpecialist": booking_specialist_agent,
        "PricingSpecialist": pricing_specialist_agent,
    }

    # Test multi-agent coordination
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[
            Message(
                role=ContentRole.USER,
                content="I need to book a flight from Los Angeles to New York for tomorrow",
            )
        ],
        current_agent_name="Coordinator",
        context={"customer_id": "12345", "session": "multi_agent_demo"},
        turn_count=0,
    )

    config = RunConfig(
        agent_registry=agent_registry, model_provider=MockModelProvider(), max_turns=10
    )

    print("🚀 Starting multi-agent coordination...")
    result = await run(initial_state, config)

    print(f"\n✅ Final Result: {result.outcome.status}")
    print(f"🔄 Total Turns: {result.final_state.turn_count}")
    print(f"👤 Final Agent: {result.final_state.current_agent_name}")

    # Demonstrate functional composition
    print("\n🔧 Functional Composition Demo:")

    # Compose search function with logging, caching, and retry
    enhanced_search = with_logging(with_cache(with_retry(search_flights_tool.execute)))

    # Test composed function
    search_args = FlightSearchArgs(origin="LAX", destination="JFK", departure_date="2024-01-15")

    result1 = await enhanced_search(search_args, {})
    print(f"First call result: {result1.status}")

    # Second call should hit cache
    result2 = await enhanced_search(search_args, {})
    print(f"Second call result: {result2.status}")

    # Test validator composition
    print("\n✅ Validator Composition Demo:")
    combined_validator = compose_validators(validate_airport_code, validate_passenger_count)

    # Test with valid data
    valid_data = FlightSearchArgs(origin="LAX", destination="JFK", departure_date="2024-01-15")
    validation_result = combined_validator(valid_data)
    print(f"Valid data validation: {validation_result}")

    # Test with invalid data
    invalid_data = FlightSearchArgs(
        origin="INVALID", destination="JFK", departure_date="2024-01-15"
    )
    validation_result = combined_validator(invalid_data)
    print(f"Invalid data validation: {validation_result}")

    print("\n🎉 Multi-agent coordination demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
