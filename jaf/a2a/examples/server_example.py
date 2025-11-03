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
from typing import Any, Dict

from pydantic import BaseModel, Field

from jaf.a2a.agent import create_a2a_agent, create_a2a_tool
from jaf.a2a.server import create_a2a_server, create_server_config

# Import A2A functionality


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
                    "tool_calls": None,
                }
            }
        elif "weather" in agent_name:
            return {
                "message": {
                    "content": f"I can check weather! You asked: '{last_message}'",
                    "tool_calls": None,
                }
            }
        else:
            return {
                "message": {
                    "content": f"Hello! I'm {agent.name}. You said: '{last_message}'",
                    "tool_calls": None,
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
    """Safe calculator tool using AST parsing"""
    import ast
    import operator

    # Allowed operations for safe calculation
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    allowed_functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
    }

    def safe_eval_node(node):
        """Safely evaluate an AST node"""
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.Name):
            # Only allow specific constants
            if node.id in ["pi", "e"]:
                import math

                return getattr(math, node.id)
            else:
                raise ValueError(f"Name '{node.id}' is not allowed")
        elif isinstance(node, ast.BinOp):
            left = safe_eval_node(node.left)
            right = safe_eval_node(node.right)
            operator_func = allowed_operators.get(type(node.op))
            if operator_func is None:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
            return operator_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval_node(node.operand)
            operator_func = allowed_operators.get(type(node.op))
            if operator_func is None:
                raise ValueError(f"Unary operator {type(node.op).__name__} is not allowed")
            return operator_func(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in allowed_functions:
                raise ValueError(f"Function '{func_name}' is not allowed")
            args = [safe_eval_node(arg) for arg in node.args]
            return allowed_functions[func_name](*args)
        else:
            raise ValueError(f"Node type {type(node).__name__} is not allowed")

    try:
        expression = args.expression.strip()

        # Additional security checks
        dangerous_patterns = [
            "import",
            "exec",
            "eval",
            "__",
            "open",
            "file",
            "input",
            "raw_input",
            "compile",
            "globals",
            "locals",
            "vars",
            "dir",
            "getattr",
            "setattr",
            "hasattr",
            "delattr",
            "callable",
        ]

        expression_lower = expression.lower()
        for pattern in dangerous_patterns:
            if pattern in expression_lower:
                return {
                    "error": f"Security violation: '{pattern}' is not allowed in expressions",
                    "result": None,
                }

        # Parse expression into AST
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return {"error": f"Invalid mathematical expression: {e!s}", "result": None}

        # Evaluate safely
        result = safe_eval_node(parsed.body)

        # Handle division by zero and other math errors
        if not isinstance(result, (int, float, complex)):
            return {"error": "Expression must evaluate to a number", "result": None}

        return {
            "result": f"The result of {expression} is {result}",
            "calculation": {"expression": expression, "result": result},
        }

    except ZeroDivisionError:
        return {"error": "Division by zero is not allowed", "result": None}
    except ValueError as e:
        return {"error": f"Invalid expression: {e!s}", "result": None}
    except Exception as e:
        return {"error": f"Calculation error: {e!s}", "result": None}


async def weather_tool(args: WeatherArgs, context) -> Dict[str, Any]:
    """Mock weather tool"""
    # Mock weather data
    weather_data = {
        "location": args.location,
        "temperature": 22 if args.units == "celsius" else 72,
        "units": args.units,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 15,
    }

    temp_unit = "°C" if args.units == "celsius" else "°F"

    return {
        "result": f"Weather in {args.location}: {weather_data['temperature']}{temp_unit}, {weather_data['condition']}",
        "weather_data": weather_data,
    }


async def translate_tool(args: TranslateArgs, context) -> Dict[str, Any]:
    """Mock translation tool"""
    # Mock translations
    translations = {
        "es": f"[Spanish] {args.text}",
        "fr": f"[French] {args.text}",
        "de": f"[German] {args.text}",
        "it": f"[Italian] {args.text}",
        "pt": f"[Portuguese] {args.text}",
    }

    translated = translations.get(
        args.target_language, f"[{args.target_language.upper()}] {args.text}"
    )

    return {
        "result": f"Translation to {args.target_language}: {translated}",
        "translation": {
            "original": args.text,
            "translated": translated,
            "target_language": args.target_language,
        },
    }


def create_example_agents():
    """Create example A2A agents with different capabilities"""

    # Math Agent with calculator tool
    calc_tool = create_a2a_tool(
        "calculator",
        "Perform mathematical calculations",
        CalculateArgs.model_json_schema(),
        calculator_tool,
    )

    math_agent = create_a2a_agent(
        "MathTutor",
        "A mathematical assistant that can solve equations and calculations",
        "You are a helpful math tutor. Use the calculator tool for computations.",
        [calc_tool],
    )

    # Weather Agent with weather tool
    weather_tool_obj = create_a2a_tool(
        "get_weather",
        "Get current weather information for a location",
        WeatherArgs.model_json_schema(),
        weather_tool,
    )

    weather_agent = create_a2a_agent(
        "WeatherBot",
        "A weather assistant that provides current weather information",
        "You are a weather assistant. Use the weather tool to get current conditions.",
        [weather_tool_obj],
    )

    # Translation Agent with translation tool
    translate_tool_obj = create_a2a_tool(
        "translate_text",
        "Translate text between languages",
        TranslateArgs.model_json_schema(),
        translate_tool,
    )

    translation_agent = create_a2a_agent(
        "Translator",
        "A multilingual assistant that can translate text between languages",
        "You are a translation assistant. Use the translation tool for language conversion.",
        [translate_tool_obj],
    )

    # General Assistant (no specific tools)
    general_agent = create_a2a_agent(
        "Assistant",
        "A general-purpose helpful assistant",
        "You are a helpful, friendly assistant ready to help with various tasks.",
        [],
    )

    return {
        "MathTutor": math_agent,
        "WeatherBot": weather_agent,
        "Translator": translation_agent,
        "Assistant": general_agent,
    }


async def main():
    """Main function to start the A2A server"""
    print("🚀 Starting A2A Server Example...")

    # Create example agents
    agents = create_example_agents()
    print(f"📦 Created {len(agents)} agents: {', '.join(agents.keys())}")

    # Create A2A server
    try:
        print("\n🌟 Starting A2A-enabled JAF server...")

        # Create server configuration
        server_config = create_server_config(
            agents=agents,
            name="JAF A2A Example Server",
            description="Multi-agent server showcasing A2A protocol capabilities",
            host="localhost",
            port=3000,
        )

        # Add mock model provider
        server_config["model_provider"] = MockModelProvider()

        print("🔧 Server configuration created")
        print("🏠 Host: localhost:3000")
        print(f"🤖 Agents: {len(agents)}")

        # Create and start the server
        server = create_a2a_server(server_config)
        await server["start"]()

        print("\n✅ Server started successfully!")
        print("\n📋 Available endpoints:")
        print("   🔍 Agent Card: http://localhost:3000/.well-known/agent-card")
        print("   🔗 A2A Endpoint: http://localhost:3000/a2a")
        print("   🏥 Health Check: http://localhost:3000/a2a/health")
        print("   ⚡ Capabilities: http://localhost:3000/a2a/capabilities")
        print("   📖 API Docs: http://localhost:3000/docs")

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
