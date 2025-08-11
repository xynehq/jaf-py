#!/usr/bin/env python3
"""
JAF Server Example - Python equivalent of the TypeScript server-demo.

This example demonstrates how to create and run a JAF server with custom agents.
"""

import asyncio
import os
from typing import Any, Dict
from pydantic import BaseModel

from jaf import (
    Agent, Tool, ToolSchema, RunConfig, 
    make_litellm_provider, run_server, generate_trace_id, generate_run_id
)
from jaf.server.types import ServerConfig
from jaf.core.tool_results import ToolResult, ToolResultStatus


# Tool argument models
class CalculatorArgs(BaseModel):
    """Arguments for the calculator tool."""
    operation: str
    a: float
    b: float


class WeatherArgs(BaseModel):
    """Arguments for the weather tool."""
    location: str


# Tool implementations
class CalculatorTool:
    """A simple calculator tool."""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="Perform basic mathematical operations (add, subtract, multiply, divide)",
            parameters=CalculatorArgs
        )
    
    async def execute(self, args: CalculatorArgs, context: Any) -> ToolResult:
        """Execute the calculation."""
        try:
            if args.operation == "add":
                result = args.a + args.b
            elif args.operation == "subtract":
                result = args.a - args.b
            elif args.operation == "multiply":
                result = args.a * args.b
            elif args.operation == "divide":
                if args.b == 0:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        error_message="Cannot divide by zero",
                        data={"operation": args.operation, "a": args.a, "b": args.b}
                    )
                result = args.a / args.b
            else:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error_message=f"Unknown operation: {args.operation}",
                    data={"operation": args.operation}
                )
            
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data=f"The result of {args.a} {args.operation} {args.b} is {result}",
                metadata={"operation": args.operation, "result": result}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_message=f"Calculation error: {str(e)}",
                data={"error": str(e)}
            )


class WeatherTool:
    """A mock weather information tool."""
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_weather",
            description="Get current weather information for a location",
            parameters=WeatherArgs
        )
    
    async def execute(self, args: WeatherArgs, context: Any) -> ToolResult:
        """Get weather information (mock implementation)."""
        # Mock weather data
        mock_weather = {
            "new york": {"temperature": "22°C", "condition": "Sunny", "humidity": "45%"},
            "london": {"temperature": "15°C", "condition": "Cloudy", "humidity": "70%"},
            "tokyo": {"temperature": "28°C", "condition": "Partly Cloudy", "humidity": "60%"},
            "paris": {"temperature": "18°C", "condition": "Rainy", "humidity": "80%"},
        }
        
        location_key = args.location.lower()
        weather = mock_weather.get(location_key, {
            "temperature": "20°C", 
            "condition": "Unknown", 
            "humidity": "50%"
        })
        
        weather_report = (
            f"Weather in {args.location.title()}:\n"
            f"Temperature: {weather['temperature']}\n"
            f"Condition: {weather['condition']}\n"
            f"Humidity: {weather['humidity']}"
        )
        
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data=weather_report,
            metadata={"location": args.location, "weather_data": weather}
        )


# Agent definitions
def create_math_agent() -> Agent:
    """Create a mathematics assistant agent."""
    
    def math_instructions(state) -> str:
        return """You are a helpful mathematics assistant. You can perform calculations using the calculator tool.
        When users ask for mathematical operations, use the calculator tool to provide accurate results.
        Always explain your calculations clearly."""
    
    return Agent(
        name="math_assistant",
        instructions=math_instructions,
        tools=[CalculatorTool()],
        output_codec=None,
        handoffs=["general_assistant"],  # Can handoff to general assistant
        model_config=None
    )


def create_weather_agent() -> Agent:
    """Create a weather information agent."""
    
    def weather_instructions(state) -> str:
        return """You are a weather information assistant. You can provide weather information for various locations.
        When users ask about weather, use the get_weather tool to fetch current conditions.
        Present the information in a friendly and informative way."""
    
    return Agent(
        name="weather_assistant",
        instructions=weather_instructions,
        tools=[WeatherTool()],
        output_codec=None,
        handoffs=["general_assistant"],  # Can handoff to general assistant
        model_config=None
    )


def create_general_agent() -> Agent:
    """Create a general purpose assistant agent."""
    
    def general_instructions(state) -> str:
        return """You are a helpful general assistant. You can help with various tasks and questions.
        If users need mathematical calculations, you can handoff to the math_assistant.
        If users need weather information, you can handoff to the weather_assistant.
        
        Available handoffs:
        - math_assistant: For mathematical calculations and problems
        - weather_assistant: For weather information and forecasts"""
    
    return Agent(
        name="general_assistant",
        instructions=general_instructions,
        tools=[],  # No direct tools, relies on handoffs
        output_codec=None,
        handoffs=["math_assistant", "weather_assistant"],
        model_config=None
    )


def create_server_config() -> ServerConfig:
    """Create the server configuration."""
    
    # Create agents
    math_agent = create_math_agent()
    weather_agent = create_weather_agent()
    general_agent = create_general_agent()
    
    agent_registry = {
        "math_assistant": math_agent,
        "weather_assistant": weather_agent,
        "general_assistant": general_agent,
    }
    
    # Create model provider
    # You can use different providers by changing the base_url
    model_provider = make_litellm_provider(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")
    )
    
    # Create run configuration
    run_config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=50,
        model_override=os.getenv("MODEL_NAME", "gpt-4o")
    )
    
    # Create server configuration
    return ServerConfig(
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        agent_registry=agent_registry,
        run_config=run_config,
        cors=True  # Enable CORS for browser access
    )


async def main():
    """Main application entry point."""
    print("🚀 Starting JAF Server Example...")
    
    # Load environment variables if .env file exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("📄 Loaded environment variables from .env file")
    except ImportError:
        print("💡 Install python-dotenv to use .env files: pip install python-dotenv")
    
    # Create and start server
    config = create_server_config()
    
    print("\n🤖 Available Agents:")
    for name, agent in config.agent_registry.items():
        print(f"  - {name}: {len(agent.tools or [])} tools, handoffs: {agent.handoffs or 'none'}")
    
    print(f"\n📡 Server starting on http://{config.host}:{config.port}")
    print("🔗 API Endpoints:")
    print(f"  - Health: http://{config.host}:{config.port}/health")
    print(f"  - Agents: http://{config.host}:{config.port}/agents")
    print(f"  - Chat: http://{config.host}:{config.port}/chat")
    print(f"  - Docs: http://{config.host}:{config.port}/docs")
    
    print("\n💬 Example chat requests:")
    print("  General: POST /chat with agent_name='general_assistant'")
    print("  Math: POST /chat with agent_name='math_assistant'")
    print("  Weather: POST /chat with agent_name='weather_assistant'")
    
    print("\n⚙️ Configuration:")
    print(f"  Model: {config.run_config.model_override or 'gpt-4o'}")
    print(f"  Max turns: {config.run_config.max_turns}")
    print(f"  CORS: {'Enabled' if config.cors else 'Disabled'}")
    
    try:
        await run_server(config)
    except KeyboardInterrupt:
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())