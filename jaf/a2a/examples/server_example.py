"""
A2A Server Example

This example demonstrates how to create and run an A2A-enabled JAF server
with multiple agents, tools, and full protocol support.

Usage:
    python server_example.py

The server will start on http://localhost:3000 with the following endpoints:
- /.well-known/agent-card - Agent discovery
- /a2a - Main A2A JSON-RPC endpoint  
- /a2a/agents/{name} - Agent-specific endpoints
- /a2a/health - Health check
- /a2a/capabilities - Server capabilities
"""

import asyncio
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Import A2A functionality
from jaf.a2a import (
    A2A, create_a2a_agent, create_a2a_tool, 
    create_server_config, start_a2a_server
)

# Mock model provider for demonstration
class MockModelProvider:
    """Mock model provider for testing"""
    
    async def get_completion(self, state, agent, config):
        # Simple mock responses based on agent type
        agent_name = agent.name.lower()
        last_message = state.messages[-1].content if state.messages else ""
        
        if "math" in agent_name:
            return {
                "message": {
                    "content": f"I can help with math! I see you said: '{last_message}'",
                    "tool_calls": None
                }
            }
        elif "weather" in agent_name:
            return {
                "message": {
                    "content": f"I can check weather! You asked: '{last_message}'",
                    "tool_calls": None
                }
            }
        else:
            return {
                "message": {
                    "content": f"Hello! I'm {agent.name}. You said: '{last_message}'",
                    "tool_calls": None
                }
            }


# Tool argument models
class CalculateArgs(BaseModel):
    """Arguments for calculator tool"""
    expression: str = Field(description="Mathematical expression to evaluate")


class WeatherArgs(BaseModel):
    """Arguments for weather tool"""
    location: str = Field(description="City or location name")
    units: str = Field(default="celsius", description="Temperature units (celsius/fahrenheit)")


class TranslateArgs(BaseModel):
    """Arguments for translation tool"""
    text: str = Field(description="Text to translate")
    target_language: str = Field(description="Target language code (e.g., 'es', 'fr', 'de')")


# Tool implementations
async def calculator_tool(args: CalculateArgs, context) -> Dict[str, Any]:
    """Safe calculator tool"""
    try:
        # Basic security check
        expression = args.expression.strip()
        if any(keyword in expression for keyword in ['import', 'exec', 'eval', '__']):
            return {
                "error": "Invalid expression: Security violation",
                "result": None
            }
        
        # Simple evaluation (in production, use a safer math parser)
        result = eval(expression)
        
        return {
            "result": f"The result of {expression} is {result}",
            "calculation": {
                "expression": expression,
                "result": result
            }
        }
    except Exception as e:
        return {
            "error": f"Calculation error: {str(e)}",
            "result": None
        }


async def weather_tool(args: WeatherArgs, context) -> Dict[str, Any]:
    """Mock weather tool"""
    # Mock weather data
    weather_data = {
        "location": args.location,
        "temperature": 22 if args.units == "celsius" else 72,
        "units": args.units,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 15
    }
    
    temp_unit = "°C" if args.units == "celsius" else "°F"
    
    return {
        "result": f"Weather in {args.location}: {weather_data['temperature']}{temp_unit}, {weather_data['condition']}",
        "weather_data": weather_data
    }


async def translate_tool(args: TranslateArgs, context) -> Dict[str, Any]:
    """Mock translation tool"""
    # Mock translations
    translations = {
        "es": f"[Spanish] {args.text}",
        "fr": f"[French] {args.text}",
        "de": f"[German] {args.text}",
        "it": f"[Italian] {args.text}",
        "pt": f"[Portuguese] {args.text}"
    }
    
    translated = translations.get(args.target_language, f"[{args.target_language.upper()}] {args.text}")
    
    return {
        "result": f"Translation to {args.target_language}: {translated}",
        "translation": {
            "original": args.text,
            "translated": translated,
            "target_language": args.target_language
        }
    }


def create_example_agents():
    """Create example A2A agents with different capabilities"""
    
    # Math Agent with calculator tool
    calc_tool = create_a2a_tool(
        "calculator",
        "Perform mathematical calculations",
        CalculateArgs.model_json_schema(),
        calculator_tool
    )
    
    math_agent = create_a2a_agent(
        "MathTutor",
        "A mathematical assistant that can solve equations and calculations",
        "You are a helpful math tutor. Use the calculator tool for computations.",
        [calc_tool]
    )
    
    # Weather Agent with weather tool
    weather_tool_obj = create_a2a_tool(
        "get_weather",
        "Get current weather information for a location",
        WeatherArgs.model_json_schema(),
        weather_tool
    )
    
    weather_agent = create_a2a_agent(
        "WeatherBot",
        "A weather assistant that provides current weather information",
        "You are a weather assistant. Use the weather tool to get current conditions.",
        [weather_tool_obj]
    )
    
    # Translation Agent with translation tool
    translate_tool_obj = create_a2a_tool(
        "translate_text",
        "Translate text between languages",
        TranslateArgs.model_json_schema(),
        translate_tool
    )
    
    translation_agent = create_a2a_agent(
        "Translator",
        "A multilingual assistant that can translate text between languages",
        "You are a translation assistant. Use the translation tool for language conversion.",
        [translate_tool_obj]
    )
    
    # General Assistant (no specific tools)
    general_agent = create_a2a_agent(
        "Assistant",
        "A general-purpose helpful assistant",
        "You are a helpful, friendly assistant ready to help with various tasks.",
        []
    )
    
    return {
        "MathTutor": math_agent,
        "WeatherBot": weather_agent, 
        "Translator": translation_agent,
        "Assistant": general_agent
    }


async def main():
    """Main function to start the A2A server"""
    print("🚀 Starting A2A Server Example...")
    
    # Create example agents
    agents = create_example_agents()
    print(f"📦 Created {len(agents)} agents: {', '.join(agents.keys())}")
    
    # Create server configuration
    server_config = create_server_config(
        agents=agents,
        name="JAF A2A Example Server",
        description="Multi-agent server showcasing A2A protocol capabilities",
        port=3000,
        host="localhost",
        version="1.0.0",
        provider={
            "organization": "JAF Framework Examples",
            "url": "https://functional-agent-framework.com"
        }
    )
    
    print("🔧 Server configuration created")
    print(f"🏠 Host: {server_config['host']}:{server_config['port']}")
    print(f"🤖 Agents: {len(server_config['agents'])}")
    
    # Start the server
    try:
        print("\n🌟 Starting A2A-enabled JAF server...")
        server = await start_a2a_server(server_config)
        
        print("\n✅ Server started successfully!")
        print("\n📋 Available endpoints:")
        print(f"   🔍 Agent Card: http://localhost:3000/.well-known/agent-card")
        print(f"   🔗 A2A Endpoint: http://localhost:3000/a2a")
        print(f"   🏥 Health Check: http://localhost:3000/a2a/health")
        print(f"   ⚡ Capabilities: http://localhost:3000/a2a/capabilities")
        print(f"   📖 API Docs: http://localhost:3000/docs")
        
        print("\n🎯 Agent-specific endpoints:")
        for agent_name in agents.keys():
            print(f"   📱 {agent_name}: http://localhost:3000/a2a/agents/{agent_name}")
        
        print("\n📝 Example A2A requests you can try:")
        print("""
    # Send message to math agent:
    curl -X POST http://localhost:3000/a2a/agents/MathTutor \\
      -H "Content-Type: application/json" \\
      -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
          "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "What is 25 * 17?"}],
            "messageId": "msg_1",
            "contextId": "test_session",
            "kind": "message"
          }
        }
      }'
    
    # Get weather:
    curl -X POST http://localhost:3000/a2a/agents/WeatherBot \\
      -H "Content-Type: application/json" \\
      -d '{
        "jsonrpc": "2.0",
        "id": "2", 
        "method": "message/send",
        "params": {
          "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "What is the weather in London?"}],
            "messageId": "msg_2",
            "contextId": "test_session",
            "kind": "message"
          }
        }
      }'
        """)
        
        print("\n🔄 Server is running... Press Ctrl+C to stop")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Received shutdown signal...")
            await server["stop"](server)
            print("✅ Server stopped gracefully")
            
    except Exception as error:
        print(f"❌ Failed to start server: {error}")
        raise


if __name__ == "__main__":
    # Run the server
    asyncio.run(main())