#!/usr/bin/env python3

import asyncio
import os
from typing import Any, Optional
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
import httpx

from jaf import Agent, create_function_tool, make_litellm_provider
from jaf.core.types import RunState, RunConfig, Message, generate_run_id, generate_trace_id
from jaf.core.engine import run


# Math tool definition
class CalculateArgs(BaseModel):
    expression: str


async def calculate(args: CalculateArgs, context: Any) -> str:
    """Safe arithmetic calculator"""
    # Security: only allow basic arithmetic
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in args.expression):
        return "Error: Invalid characters in expression"

    try:
        result = eval(args.expression)
        return f"{args.expression} = {result}"
    except Exception:
        return "Error: Invalid expression"


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

        print(f"[DEBUG] Fetching coffee data from: {url}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)  # httpx.response
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

            result = f"Found {len(coffee_data)} {args.coffee_type} coffee options. Here are the first 3:\n\n"
            result += "\n".join(formatted_coffees)

            if len(coffee_data) > 3:
                result += f"\n\n... and {len(coffee_data) - 3} more options available!"

            return result  # str
        else:
            return f"No {args.coffee_type} coffee data found"

    except httpx.TimeoutException:
        return "Error: Coffee API request timed out"
    except httpx.HTTPStatusError as e:
        return f"Error: Coffee API returned status"
    except Exception as e:
        return f"Error fetching coffee data: {str(e)}"


# Create tools
math_tool = create_function_tool(
    {
        "name": "calculate",
        "description": "Perform arithmetic calculations",
        "execute": calculate,
        "parameters": CalculateArgs,
    }
)

coffee_tool = create_function_tool(
    {
        "name": "get_coffee_info",
        "description": 'Get information about coffee types from the API. Use "hot" for hot coffee or "iced" for iced coffee.',
        "execute": get_coffee_data,
        "parameters": CoffeeArgs,
    }
)

# Create agent with both tools
multi_agent = Agent(
    name="MultiAgent",
    instructions=lambda state: """You are a helpful assistant with access to math calculations and coffee information.

For math questions:
- Always use the calculate tool for any arithmetic expressions
- Example: "Calculate 15 * 7" -> call calculate tool with expression "15 * 7"

For coffee questions:
- Use the get_coffee_info tool to fetch coffee data
- You can get "hot" coffee (default) or "iced" coffee information
- Example: "Show me hot coffee options" -> call get_coffee_info tool with coffee_type "hot"

Always use the appropriate tool for each request.""",
    tools=[math_tool, coffee_tool],
)


# FastAPI server
app = FastAPI(title="JAF Math Server", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(request: dict):
    """Chat with the math agent"""
    try:
        # Get message from request
        user_message = request.get("message", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Create LiteLLM provider with specific model that supports tool calling
        model_provider = make_litellm_provider("BASE_URL", "API_URL")

        print(f"[DEBUG] Processing message: {user_message}")

        # Create run state
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(role="user", content=user_message)],
            current_agent_name="MultiAgent",
            context={},
            turn_count=0,
        )

        # Create run config with model override for tool calling
        config = RunConfig(
            agent_registry={"MultiAgent": multi_agent},
            model_provider=model_provider,
            max_turns=5,
            model_override="gemini-2.5-pro",  # Ensure we use a model that supports tool calling
        )

        print(f"[DEBUG] Starting agent")

        # Run the agent
        result = await run(initial_state, config)

        print(
            f"[DEBUG] Agent run completed. Final state has {len(result.final_state.messages)} messages"
        )
        print(f"[DEBUG] Outcome: {result.outcome}")

        # Debug: Print all messages
        for i, msg in enumerate(result.final_state.messages):
            print(f"[DEBUG] Message {i}: role={msg.role}, content={msg.content[:100]}...")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"[DEBUG] Tool calls: {msg.tool_calls}")

        # Extract response - get the last assistant message
        assistant_messages = [msg for msg in result.final_state.messages if msg.role == "assistant"]
        if assistant_messages:
            response = assistant_messages[-1].content
        else:
            response = "No assistant response generated"

        return {
            "response": response,
            "status": "success",
            "debug": {
                "total_messages": len(result.final_state.messages),
                "turn_count": result.final_state.turn_count,
                "outcome": str(result.outcome),
            },
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
