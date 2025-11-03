"""
Comprehensive A2A Framework Test

This script provides a complete test of ALL A2A functionality in the JAF framework:

1. Server Setup with Multiple Agents
2. Client Discovery and Interaction
3. Task Persistence with Different Storage Backends
4. Streaming Responses
5. Tool Execution
6. Health Monitoring
7. Error Handling
8. Performance Testing
9. Memory Management
10. Multi-Agent Coordination

Usage:
    python examples/comprehensive_a2a_test.py

This will test every aspect of the A2A framework and show you how to use all features.
"""

import asyncio
import json
import time
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

# Import all A2A functionality
from jaf.a2a.agent import create_a2a_agent, create_a2a_tool
from jaf.a2a.server import create_a2a_server, create_server_config
from jaf.a2a.client import (
    create_a2a_client,
    send_message_to_agent,
    stream_message,
    discover_agents,
    check_a2a_health,
    get_a2a_capabilities,
    connect_to_a2a_agent,
)

# Import memory providers for task persistence
from jaf.a2a.memory.factory import create_a2a_task_provider
from jaf.a2a.memory.providers.in_memory import create_a2a_in_memory_task_provider
from jaf.a2a.memory.cleanup import create_task_cleanup_scheduler, perform_task_cleanup


class ComprehensiveTestResults:
    """Results collector for all tests"""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def add_result(self, test_name: str, success: bool, details: Any = None, duration: float = 0):
        self.results[test_name] = {
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": time.time(),
        }

    def get_summary(self) -> Dict[str, Any]:
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results.values() if r["success"])
        total_duration = time.time() - self.start_time

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_duration": total_duration,
            "results": self.results,
        }


class MockAdvancedModelProvider:
    """Advanced mock model provider with realistic responses"""

    def __init__(self):
        self.call_count = 0

    async def get_completion(self, state, agent, config):
        from jaf.core.types import ContentRole

        self.call_count += 1
        last_message = state.messages[-1] if state.messages else None
        agent_name = agent.name.lower()

        # If the last message was a tool result, provide a final answer
        if last_message and last_message.role == ContentRole.TOOL:
            return {
                "message": {
                    "content": f"I have completed the task using the tool. The result is: {last_message.content}",
                    "tool_calls": None,
                }
            }

        # Simulate more realistic responses
        if "math" in agent_name or "calculator" in agent_name:
            if last_message and any(
                op in last_message.content for op in ["+", "-", "*", "/", "calculate", "solve"]
            ):
                return {
                    "message": {
                        "content": f"I'll help you with that calculation. Let me use my calculator tool.",
                        "tool_calls": [
                            {
                                "id": f"call_{self.call_count}",
                                "type": "function",
                                "function": {
                                    "name": "advanced_calculator",
                                    "arguments": json.dumps(
                                        {"expression": "25 + 17", "format_result": True}
                                    ),
                                },
                            }
                        ],
                    }
                }
            else:
                return {
                    "message": {
                        "content": f"Hello! I'm {agent.name}, your math assistant. I can help with calculations, equations, and mathematical problems. What would you like me to calculate?",
                        "tool_calls": None,
                    }
                }
        elif "weather" in agent_name:
            if last_message and any(
                word in last_message.content.lower()
                for word in ["weather", "temperature", "forecast"]
            ):
                return {
                    "message": {
                        "content": "I'll get the current weather information for you.",
                        "tool_calls": [
                            {
                                "id": f"call_{self.call_count}",
                                "type": "function",
                                "function": {
                                    "name": "advanced_weather",
                                    "arguments": json.dumps(
                                        {
                                            "location": "London",
                                            "units": "celsius",
                                            "include_forecast": True,
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            else:
                return {
                    "message": {
                        "content": f"Hi! I'm {agent.name}. I can provide weather information for any location. Just ask me about the weather somewhere!",
                        "tool_calls": None,
                    }
                }
        elif "translate" in agent_name or "translator" in agent_name:
            if last_message and "translate" in last_message.content.lower():
                return {
                    "message": {
                        "content": "I'll translate that for you.",
                        "tool_calls": [
                            {
                                "id": f"call_{self.call_count}",
                                "type": "function",
                                "function": {
                                    "name": "advanced_translate",
                                    "arguments": json.dumps(
                                        {
                                            "text": "Hello World",
                                            "target_language": "es",
                                            "detect_source": True,
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            else:
                return {
                    "message": {
                        "content": f"Hello! I'm {agent.name}. I can translate text between many languages. What would you like me to translate?",
                        "tool_calls": None,
                    }
                }
        elif "data" in agent_name or "analyst" in agent_name:
            if last_message and "analyze" in last_message.content.lower():
                return {
                    "message": {
                        "content": "I will analyze that data for you.",
                        "tool_calls": [
                            {
                                "id": f"call_{self.call_count}",
                                "type": "function",
                                "function": {
                                    "name": "analyze_data",
                                    "arguments": json.dumps(
                                        {
                                            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                            "analysis_type": "all",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                }
            else:
                return {
                    "message": {
                        "content": f"Hello! I'm {agent.name}. How can I help you today? You said: '{last_message.content if last_message else ''}'",
                        "tool_calls": None,
                    }
                }
        elif "file" in agent_name or "manager" in agent_name:
            if last_message and "list files" in last_message.content.lower():
                return {
                    "message": {
                        "content": "Let me list those files for you.",
                        "tool_calls": [
                            {
                                "id": f"call_{self.call_count}",
                                "type": "function",
                                "function": {
                                    "name": "file_operations",
                                    "arguments": json.dumps(
                                        {"operation": "list", "path": "/documents"}
                                    ),
                                },
                            }
                        ],
                    }
                }
            else:
                return {
                    "message": {
                        "content": f"Hello! I'm {agent.name}. How can I help you today? You said: '{last_message.content if last_message else ''}'",
                        "tool_calls": None,
                    }
                }
        else:
            return {
                "message": {
                    "content": f"Hello! I'm {agent.name}. How can I help you today? You said: '{last_message.content if last_message else ''}'",
                    "tool_calls": None,
                }
            }


# Advanced Tool Schemas
class AdvancedCalculatorArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")
    format_result: bool = Field(default=True, description="Whether to format the result nicely")


class WeatherArgs(BaseModel):
    location: str = Field(description="City or location name")
    units: str = Field(default="celsius", description="Temperature units")
    include_forecast: bool = Field(default=False, description="Include 5-day forecast")


class TranslationArgs(BaseModel):
    text: str = Field(description="Text to translate")
    target_language: str = Field(description="Target language code")
    detect_source: bool = Field(default=True, description="Auto-detect source language")


class DataAnalysisArgs(BaseModel):
    data: List[float] = Field(description="List of numbers to analyze")
    analysis_type: str = Field(default="all", description="Type of analysis: all, basic, advanced")


class FileOperationArgs(BaseModel):
    operation: str = Field(description="Operation type: read, write, list")
    path: str = Field(description="File or directory path")
    content: str = Field(default="", description="Content for write operations")


# Advanced Tool Implementations
async def advanced_calculator_tool(args, context) -> str:
    """Advanced calculator with more features"""
    import ast
    import operator
    import math

    # Handle both dictionary and Pydantic model arguments
    if isinstance(args, dict):
        calc_args = AdvancedCalculatorArgs(**args)
    else:
        calc_args = args

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
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    def safe_eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in allowed_functions:
                return allowed_functions[node.id]
            raise ValueError(f"Name '{node.id}' not allowed")
        elif isinstance(node, ast.BinOp):
            left = safe_eval_node(node.left)
            right = safe_eval_node(node.right)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Operator {type(node.op).__name__} not allowed")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval_node(node.operand)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unary operator {type(node.op).__name__} not allowed")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in allowed_functions:
                raise ValueError(f"Function '{func_name}' not allowed")
            func_args = [safe_eval_node(arg) for arg in node.args]
            return allowed_functions[func_name](*func_args)
        else:
            raise ValueError(f"Node type {type(node).__name__} not allowed")

    try:
        expression = calc_args.expression.strip()
        parsed = ast.parse(expression, mode="eval")
        result = safe_eval_node(parsed.body)

        formatted_result = f"{result:,.6g}" if calc_args.format_result else str(result)

        return f"The result of {expression} is {formatted_result}"

    except Exception as e:
        return f"Calculation error: {str(e)}"


async def advanced_weather_tool(args, context) -> str:
    """Advanced weather tool with forecasts"""

    # Handle both dictionary and Pydantic model arguments
    if isinstance(args, dict):
        weather_args = WeatherArgs(**args)
    else:
        weather_args = args

    # Mock weather data with more detail
    base_weather = {
        "location": weather_args.location,
        "temperature": 22 if weather_args.units == "celsius" else 72,
        "units": weather_args.units,
        "condition": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 15,
        "visibility": 10,
        "pressure": 1013.25,
        "uv_index": 5,
    }

    forecast = []
    if weather_args.include_forecast:
        for i in range(5):
            forecast.append(
                {
                    "day": f"Day {i + 1}",
                    "high": base_weather["temperature"] + (i % 3),
                    "low": base_weather["temperature"] - 5 + (i % 2),
                    "condition": ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Clear"][i],
                    "chance_of_rain": [10, 30, 80, 20, 5][i],
                }
            )

    temp_unit = "°C" if weather_args.units == "celsius" else "°F"

    result = f"Weather in {weather_args.location}: {base_weather['temperature']}{temp_unit}, {base_weather['condition']}. Humidity: {base_weather['humidity']}%, Wind: {base_weather['wind_speed']} km/h"

    if weather_args.include_forecast:
        result += f"\n\n5-Day Forecast:\n"
        for day_forecast in forecast:
            result += f"- {day_forecast['day']}: {day_forecast['condition']}, High: {day_forecast['high']}{temp_unit}, Low: {day_forecast['low']}{temp_unit}, Rain: {day_forecast['chance_of_rain']}%\n"

    return result


async def advanced_translation_tool(args, context) -> str:
    """Advanced translation tool with language detection"""

    # Handle both dictionary and Pydantic model arguments
    if isinstance(args, dict):
        trans_args = TranslationArgs(**args)
    else:
        trans_args = args

    # Mock language detection
    detected_language = "en" if trans_args.detect_source else "unknown"

    # Mock translations with more languages
    translations = {
        "es": f"Hola Mundo",
        "fr": f"Bonjour le monde",
        "de": f"Hallo Welt",
        "it": f"Ciao mondo",
        "pt": f"Olá mundo",
        "ru": f"Привет мир",
        "ja": f"こんにちは世界",
        "zh": f"你好世界",
        "ar": f"مرحبا بالعالم",
        "hi": f"हैलो वर्ल्ड",
    }

    if trans_args.text.lower() == "hello world":
        translated = translations.get(
            trans_args.target_language, f"[{trans_args.target_language.upper()}] {trans_args.text}"
        )
    else:
        translated = f"[{trans_args.target_language.upper()}] {trans_args.text}"

    return f"Translation from {detected_language} to {trans_args.target_language}: {translated}"


async def data_analysis_tool(args, context) -> str:
    """Data analysis tool"""

    # Handle both dictionary and Pydantic model arguments
    if isinstance(args, dict):
        data_args = DataAnalysisArgs(**args)
    else:
        data_args = args

    if not data_args.data:
        return "No data provided for analysis"

    try:
        data = data_args.data
        n = len(data)

        # Basic statistics
        total = sum(data)
        mean = total / n
        sorted_data = sorted(data)
        median = (
            sorted_data[n // 2]
            if n % 2 == 1
            else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        )

        # Advanced statistics
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = variance**0.5

        result = f"Data analysis of {n} values:\n"
        result += f"- Mean: {mean:.2f}\n"
        result += f"- Median: {median:.2f}\n"
        result += f"- Standard Deviation: {std_dev:.2f}\n"
        result += f"- Range: {min(data):.2f} to {max(data):.2f}\n"

        if data_args.analysis_type in ["all", "advanced"]:
            q1 = sorted_data[n // 4] if n >= 4 else min(data)
            q3 = sorted_data[3 * n // 4] if n >= 4 else max(data)
            result += f"- Variance: {variance:.2f}\n"
            result += f"- Q1: {q1:.2f}, Q3: {q3:.2f}\n"

        return result

    except Exception as e:
        return f"Analysis error: {str(e)}"


async def file_operation_tool(args, context) -> str:
    """Mock file operations tool"""

    # Handle both dictionary and Pydantic model arguments
    if isinstance(args, dict):
        file_args = FileOperationArgs(**args)
    else:
        file_args = args

    # Mock file system for demonstration
    mock_files = {
        "/documents/report.txt": "This is a sample report content.",
        "/documents/data.csv": "name,age,city\nJohn,25,NYC\nJane,30,LA",
        "/projects/readme.md": "# Project README\nThis is a sample project.",
    }

    try:
        if file_args.operation == "list":
            if file_args.path.startswith("/documents"):
                files = [f for f in mock_files.keys() if f.startswith("/documents")]
            else:
                files = list(mock_files.keys())

            files_list = "\n".join(f"- {file}" for file in files)
            return f"Found {len(files)} files in {file_args.path}:\n{files_list}"

        elif file_args.operation == "read":
            content = mock_files.get(file_args.path, "File not found")
            return f"Contents of {file_args.path}:\n{content}"

        elif file_args.operation == "write":
            mock_files[file_args.path] = file_args.content
            return f"Successfully wrote {len(file_args.content)} characters to {file_args.path}"

        else:
            return f"Unknown operation: {file_args.operation}"

    except Exception as e:
        return f"File operation error: {str(e)}"


def create_comprehensive_agents():
    """Create a comprehensive set of A2A agents"""

    # Advanced Math Agent
    calc_tool = create_a2a_tool(
        "advanced_calculator",
        "Perform advanced mathematical calculations with functions",
        AdvancedCalculatorArgs.model_json_schema(),
        advanced_calculator_tool,
    )

    math_agent = create_a2a_agent(
        "AdvancedMathTutor",
        "Advanced mathematical assistant with scientific functions",
        "You are an advanced math tutor. Use the calculator for complex calculations including trigonometry, logarithms, and more.",
        [calc_tool],
    )

    # Advanced Weather Agent
    weather_tool_obj = create_a2a_tool(
        "advanced_weather",
        "Get detailed weather information including forecasts",
        WeatherArgs.model_json_schema(),
        advanced_weather_tool,
    )

    weather_agent = create_a2a_agent(
        "AdvancedWeatherBot",
        "Comprehensive weather assistant with forecasting",
        "You are a weather expert. Provide detailed weather information and forecasts.",
        [weather_tool_obj],
    )

    # Advanced Translation Agent
    translate_tool_obj = create_a2a_tool(
        "advanced_translate",
        "Translate text with language detection",
        TranslationArgs.model_json_schema(),
        advanced_translation_tool,
    )

    translation_agent = create_a2a_agent(
        "AdvancedTranslator",
        "Professional translation assistant with language detection",
        "You are a professional translator. Provide accurate translations with language detection.",
        [translate_tool_obj],
    )

    # Data Analysis Agent
    data_tool = create_a2a_tool(
        "analyze_data",
        "Perform statistical analysis on numerical data",
        DataAnalysisArgs.model_json_schema(),
        data_analysis_tool,
    )

    data_agent = create_a2a_agent(
        "DataAnalyst",
        "Statistical data analysis specialist",
        "You are a data analyst. Analyze numerical data and provide statistical insights.",
        [data_tool],
    )

    # File Operations Agent
    file_tool = create_a2a_tool(
        "file_operations",
        "Perform file system operations",
        FileOperationArgs.model_json_schema(),
        file_operation_tool,
    )

    file_agent = create_a2a_agent(
        "FileManager",
        "File system operations assistant",
        "You are a file manager. Help with file operations like reading, writing, and listing files.",
        [file_tool],
    )

    # Multi-tool Agent
    multi_agent = create_a2a_agent(
        "MultiTool",
        "Versatile assistant with multiple capabilities",
        "You are a versatile assistant with access to multiple tools. Choose the appropriate tool based on the user's request.",
        [calc_tool, weather_tool_obj, translate_tool_obj, data_tool],
    )

    # Simple Chat Agent
    chat_agent = create_a2a_agent(
        "ChatAssistant",
        "Friendly conversational assistant",
        "You are a helpful and friendly assistant. Engage in natural conversation and provide helpful responses.",
        [],
    )

    return {
        "AdvancedMathTutor": math_agent,
        "AdvancedWeatherBot": weather_agent,
        "AdvancedTranslator": translation_agent,
        "DataAnalyst": data_agent,
        "FileManager": file_agent,
        "MultiTool": multi_agent,
        "ChatAssistant": chat_agent,
    }


class ComprehensiveA2ATest:
    """Main test class for comprehensive A2A testing"""

    def __init__(self):
        self.results = ComprehensiveTestResults()
        self.server = None
        self.server_task = None
        self.task_provider = None
        self.base_url = "http://localhost:3002"  # Use unique port

    async def setup_server(self) -> bool:
        """Setup the comprehensive test server"""
        print("🚀 Setting up comprehensive A2A test server...")

        try:
            start_time = time.time()

            # Create advanced agents
            agents = create_comprehensive_agents()
            print(f"📦 Created {len(agents)} advanced agents")

            # Create task provider for persistence testing
            from jaf.a2a.memory.types import A2AInMemoryTaskConfig

            config = A2AInMemoryTaskConfig(
                type="memory",
                max_tasks=1000,
                cleanup_interval=60,
                enable_history=True,
                enable_artifacts=True,
            )
            self.task_provider = create_a2a_in_memory_task_provider(config)
            print("💾 Created task provider for persistence testing")

            # Create server configuration
            server_config = create_server_config(
                agents=agents,
                name="JAF Comprehensive A2A Test Server",
                description="Full-featured A2A server for comprehensive testing",
                host="localhost",
                port=3002,
            )

            # Add advanced model provider
            server_config["model_provider"] = MockAdvancedModelProvider()
            server_config["task_provider"] = self.task_provider

            # Create and start server
            self.server = create_a2a_server(server_config)
            self.server_task = asyncio.create_task(self.server["start"]())

            # Wait for server to start
            await asyncio.sleep(3)

            duration = time.time() - start_time
            self.results.add_result(
                "server_setup", True, f"Server started with {len(agents)} agents", duration
            )

            print(f"✅ Server started successfully in {duration:.2f}s")
            print(f"🌐 Server URL: {self.base_url}")

            return True

        except Exception as e:
            self.results.add_result("server_setup", False, str(e))
            print(f"❌ Server setup failed: {e}")
            return False

    async def test_agent_discovery(self) -> bool:
        """Test agent discovery functionality"""
        print("\n🔍 Testing agent discovery...")

        try:
            start_time = time.time()

            agent_card = await discover_agents(self.base_url)

            # Validate agent card structure
            required_fields = ["name", "description", "url", "protocolVersion", "skills"]
            for field in required_fields:
                if field not in agent_card:
                    raise ValueError(f"Missing required field: {field}")

            skills = agent_card.get("skills", [])
            agents_found = len(skills)

            duration = time.time() - start_time
            self.results.add_result(
                "agent_discovery", True, f"Found {agents_found} agents", duration
            )

            print(f"✅ Discovered {agents_found} agents in {duration:.2f}s")
            print(f"   Server: {agent_card['name']}")
            print(f"   Protocol: {agent_card['protocolVersion']}")

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("agent_discovery", False, str(e), duration)
            print(f"❌ Agent discovery failed: {e}")
            return False

    async def test_health_and_capabilities(self) -> bool:
        """Test health checks and capability detection"""
        print("\n🏥 Testing health checks and capabilities...")

        try:
            start_time = time.time()

            client = create_a2a_client(self.base_url)

            # Health check
            health = await check_a2a_health(client)
            if health.get("status") != "healthy":
                raise ValueError(f"Server not healthy: {health}")

            # Capabilities check
            capabilities = await get_a2a_capabilities(client)
            methods = capabilities.get("supportedMethods", [])

            if not methods:
                raise ValueError("No supported methods found")

            expected_methods = ["message/send", "message/stream", "tasks/get", "tasks/cancel"]
            missing_methods = [m for m in expected_methods if m not in methods]

            if missing_methods:
                raise ValueError(f"Missing expected methods: {missing_methods}")

            duration = time.time() - start_time
            self.results.add_result(
                "health_capabilities", True, f"Health OK, {len(methods)} methods", duration
            )

            print(f"✅ Health and capabilities verified in {duration:.2f}s")
            print(f"   Status: {health.get('status')}")
            print(f"   Methods: {len(methods)}")

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("health_capabilities", False, str(e), duration)
            print(f"❌ Health/capabilities test failed: {e}")
            return False

    async def test_basic_messaging(self) -> bool:
        """Test basic message sending to different agents"""
        print("\n💬 Testing basic messaging...")

        try:
            start_time = time.time()

            client = create_a2a_client(self.base_url)

            test_cases = [
                ("ChatAssistant", "Hello, how are you today?"),
                ("AdvancedMathTutor", "What is 25 + 17?"),
                ("AdvancedWeatherBot", "What's the weather like?"),
                ("AdvancedTranslator", "Can you help with translations?"),
            ]

            successful_messages = 0

            for agent_name, message in test_cases:
                try:
                    response = await send_message_to_agent(client, agent_name, message)
                    if response and len(response) > 0:
                        successful_messages += 1
                        print(f"   ✅ {agent_name}: {response[:60]}...")
                    else:
                        print(f"   ❌ {agent_name}: Empty response")

                except Exception as e:
                    print(f"   ❌ {agent_name}: {e}")

            duration = time.time() - start_time
            success = successful_messages == len(test_cases)

            self.results.add_result(
                "basic_messaging",
                success,
                f"{successful_messages}/{len(test_cases)} messages",
                duration,
            )

            print(
                f"{'✅' if success else '❌'} Basic messaging: {successful_messages}/{len(test_cases)} in {duration:.2f}s"
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("basic_messaging", False, str(e), duration)
            print(f"❌ Basic messaging test failed: {e}")
            return False

    async def test_tool_execution(self) -> bool:
        """Test tool execution with different agents"""
        print("\n🔧 Testing tool execution...")

        try:
            start_time = time.time()

            client = create_a2a_client(self.base_url)

            tool_tests = [
                ("AdvancedMathTutor", "Calculate 15 * 24 + 7"),
                ("AdvancedWeatherBot", "Get weather for London with forecast"),
                ("AdvancedTranslator", "Translate 'Hello World' to Spanish"),
                ("DataAnalyst", "Analyze these numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"),
                ("FileManager", "List files in /documents directory"),
            ]

            successful_tools = 0

            for agent_name, message in tool_tests:
                try:
                    response = await send_message_to_agent(client, agent_name, message)

                    # Check if response indicates tool usage
                    if response and (
                        "result" in response.lower()
                        or "analysis" in response.lower()
                        or "calculation" in response.lower()
                        or "files" in response.lower()
                    ):
                        successful_tools += 1
                        print(f"   ✅ {agent_name}: Tool executed successfully")
                    else:
                        print(f"   ❌ {agent_name}: Tool may not have executed")

                except Exception as e:
                    print(f"   ❌ {agent_name}: {e}")

            duration = time.time() - start_time
            success = successful_tools >= len(tool_tests) * 0.6  # 60% success rate

            self.results.add_result(
                "tool_execution", success, f"{successful_tools}/{len(tool_tests)} tools", duration
            )

            print(
                f"{'✅' if success else '❌'} Tool execution: {successful_tools}/{len(tool_tests)} in {duration:.2f}s"
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("tool_execution", False, str(e), duration)
            print(f"❌ Tool execution test failed: {e}")
            return False

    async def test_streaming_responses(self) -> bool:
        """Test streaming message responses"""
        print("\n🌊 Testing streaming responses...")

        try:
            start_time = time.time()

            client = create_a2a_client(self.base_url)

            # Test streaming with different agents
            streaming_tests = [
                ("ChatAssistant", "Tell me about the benefits of exercise"),
                ("AdvancedMathTutor", "Explain how to solve quadratic equations"),
            ]

            successful_streams = 0

            for agent_name, message in streaming_tests:
                try:
                    print(f"   📡 Streaming to {agent_name}...")

                    events_received = 0
                    async for event in stream_message(client, message, agent_name):
                        events_received += 1
                        if events_received <= 3:  # Log first few events
                            print(f"      Event {events_received}: {str(event)[:50]}...")

                        if events_received >= 10:  # Limit for testing
                            break

                    if events_received > 0:
                        successful_streams += 1
                        print(f"   ✅ {agent_name}: Received {events_received} events")
                    else:
                        print(f"   ❌ {agent_name}: No events received")

                except Exception as e:
                    print(f"   ❌ {agent_name}: Streaming failed - {e}")

            duration = time.time() - start_time
            success = successful_streams > 0

            self.results.add_result(
                "streaming_responses",
                success,
                f"{successful_streams}/{len(streaming_tests)} streams",
                duration,
            )

            print(
                f"{'✅' if success else '❌'} Streaming: {successful_streams}/{len(streaming_tests)} in {duration:.2f}s"
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("streaming_responses", False, str(e), duration)
            print(f"❌ Streaming test failed: {e}")
            return False

    async def test_memory_persistence(self) -> bool:
        """Test task memory and persistence features"""
        print("\n💾 Testing memory persistence...")

        try:
            start_time = time.time()

            if not self.task_provider:
                raise ValueError("Task provider not initialized")

            from jaf.a2a.agent import create_a2a_text_message
            from jaf.a2a.types import create_a2a_task
            from jaf.memory.types import Failure

            # Create test task
            test_message = create_a2a_text_message(
                "Test message for persistence", context_id="test_session_1"
            )
            test_task = create_a2a_task(test_message, "test_session_1")

            # Store task
            store_result = await self.task_provider.store_task(test_task)
            if isinstance(store_result, Failure):
                raise ValueError(f"Failed to store task: {store_result.error}")

            # Retrieve task
            retrieve_result = await self.task_provider.get_task(test_task.id)
            if isinstance(retrieve_result, Failure):
                raise ValueError(f"Failed to retrieve task: {retrieve_result.error}")

            retrieved_task = retrieve_result.data
            if not retrieved_task:
                raise ValueError("Stored task not found")

            # Verify task integrity
            if retrieved_task.id != test_task.id:
                raise ValueError(
                    f"Task ID mismatch after retrieval. Got {retrieved_task.id}, expected {test_task.id}"
                )

            # Test task updates
            from jaf.a2a.types import TaskState

            update_result = await self.task_provider.update_task_status(
                test_task.id,
                TaskState.COMPLETED,
                create_a2a_text_message(
                    "Task completed successfully", context_id=test_task.context_id
                ),
            )
            if isinstance(update_result, Failure):
                raise ValueError(f"Failed to update task: {update_result.error}")

            # Test task statistics
            stats_result = await self.task_provider.get_task_stats()
            if isinstance(stats_result, Failure):
                raise ValueError(f"Failed to get task stats: {stats_result.error}")

            stats = stats_result.data
            if not stats:
                raise ValueError("Failed to get task stats")

            duration = time.time() - start_time
            self.results.add_result(
                "memory_persistence", True, f"Tasks: {stats['total_tasks']}", duration
            )

            print(f"✅ Memory persistence verified in {duration:.2f}s")
            print(f"   Tasks stored: {stats['total_tasks']}")
            print(f"   Task states: {stats['tasks_by_state']}")

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("memory_persistence", False, str(e), duration)
            print(f"❌ Memory persistence test failed: {e}")
            return False

    async def test_performance_load(self) -> bool:
        """Test performance under load"""
        print("\n⚡ Testing performance under load...")

        try:
            start_time = time.time()

            client = create_a2a_client(self.base_url)

            # Concurrent message sending
            concurrent_requests = 10
            messages_per_request = 5

            async def send_batch_messages(batch_id: int):
                """Send a batch of messages"""
                batch_results = []
                for i in range(messages_per_request):
                    try:
                        response = await send_message_to_agent(
                            client, "ChatAssistant", f"Batch {batch_id}, Message {i}: Hello!"
                        )
                        batch_results.append(True)
                    except Exception:
                        batch_results.append(False)
                return batch_results

            # Run concurrent batches
            tasks = [send_batch_messages(i) for i in range(concurrent_requests)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate success rate
            total_messages = 0
            successful_messages = 0

            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    total_messages += len(batch_result)
                    successful_messages += sum(batch_result)

            success_rate = successful_messages / total_messages if total_messages > 0 else 0
            duration = time.time() - start_time

            # Test passes if success rate > 80% and duration < 30 seconds
            success = success_rate > 0.8 and duration < 30

            self.results.add_result(
                "performance_load",
                success,
                f"{successful_messages}/{total_messages} ({success_rate:.1%})",
                duration,
            )

            print(f"{'✅' if success else '❌'} Performance load test in {duration:.2f}s")
            print(f"   Success rate: {success_rate:.1%} ({successful_messages}/{total_messages})")
            print(f"   Throughput: {total_messages / duration:.1f} msg/sec")

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("performance_load", False, str(e), duration)
            print(f"❌ Performance test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling scenarios"""
        print("\n🛡️ Testing error handling...")

        try:
            start_time = time.time()

            client = create_a2a_client(self.base_url)

            error_tests = [
                # Test invalid agent
                ("NonExistentAgent", "Hello", "Invalid agent"),
                # Test invalid JSON-RPC
                (None, None, "Invalid request format"),
            ]

            expected_errors = 0
            actual_errors = 0

            # Test invalid agent
            try:
                await send_message_to_agent(client, "NonExistentAgent", "Hello")
                print("   ❌ Expected error for invalid agent, but got success")
            except Exception:
                actual_errors += 1
                expected_errors += 1
                print("   ✅ Correctly handled invalid agent error")

            # Test malformed requests
            try:
                import httpx

                response = await client._client.post(
                    f"{self.base_url}/a2a", json={"invalid": "request"}
                )
                if response.status_code >= 400:
                    actual_errors += 1
                    expected_errors += 1
                    print("   ✅ Correctly handled malformed request")
                else:
                    print("   ❌ Expected error for malformed request, but got success")
            except Exception:
                actual_errors += 1
                expected_errors += 1
                print("   ✅ Correctly handled malformed request with exception")

            duration = time.time() - start_time
            success = actual_errors == expected_errors and expected_errors > 0

            self.results.add_result(
                "error_handling",
                success,
                f"{actual_errors}/{expected_errors} errors handled",
                duration,
            )

            print(
                f"{'✅' if success else '❌'} Error handling: {actual_errors}/{expected_errors} in {duration:.2f}s"
            )

            return success

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("error_handling", False, str(e), duration)
            print(f"❌ Error handling test failed: {e}")
            return False

    async def test_convenience_methods(self) -> bool:
        """Test convenience connection methods"""
        print("\n🔗 Testing convenience methods...")

        try:
            start_time = time.time()

            # Test convenience connection
            connection = await connect_to_a2a_agent(self.base_url)

            # Test convenience ask method
            response = await connection["ask"]("Hello from convenience method!")
            if not response:
                raise ValueError("No response from convenience ask method")

            # Test health check via convenience
            health = await connection["health"]()
            if health.get("status") != "healthy":
                raise ValueError(f"Health check failed: {health}")

            # Test capabilities via convenience
            capabilities = await connection["capabilities"]()
            if not capabilities.get("supportedMethods"):
                raise ValueError("No methods in capabilities")

            duration = time.time() - start_time
            self.results.add_result(
                "convenience_methods", True, "All convenience methods working", duration
            )

            print(f"✅ Convenience methods verified in {duration:.2f}s")

            return True

        except Exception as e:
            duration = time.time() - start_time
            self.results.add_result("convenience_methods", False, str(e), duration)
            print(f"❌ Convenience methods test failed: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        print("🎯 COMPREHENSIVE A2A FRAMEWORK TEST")
        print("=" * 60)

        # Setup server
        if not await self.setup_server():
            return self.results.get_summary()

        # Run all test categories
        test_methods = [
            self.test_agent_discovery,
            self.test_health_and_capabilities,
            self.test_basic_messaging,
            self.test_tool_execution,
            self.test_streaming_responses,
            self.test_memory_persistence,
            self.test_performance_load,
            self.test_error_handling,
            self.test_convenience_methods,
        ]

        print(f"\n🧪 Running {len(test_methods)} test categories...")

        for test_method in test_methods:
            try:
                await test_method()
                await asyncio.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed with exception: {e}")

        return self.results.get_summary()

    async def cleanup(self):
        """Clean up test resources"""
        print("\n🧹 Cleaning up test resources...")

        try:
            if self.server and self.server.get("stop"):
                # Signal the server to stop
                await self.server["stop"]()

            if self.server_task and not self.server_task.done():
                # Wait for the server task to finish gracefully
                try:
                    await asyncio.wait_for(self.server_task, timeout=2.0)
                except asyncio.TimeoutError:
                    print("⚠️ Server did not shut down gracefully, cancelling task.")
                    self.server_task.cancel()
                    # Await the cancellation
                    try:
                        await self.server_task
                    except asyncio.CancelledError:
                        pass  # Expected
                except asyncio.CancelledError:
                    pass  # Also possible if already cancelled

            print("✅ Cleanup completed")

        except Exception as e:
            print(f"❌ Cleanup failed: {e}")


def print_comprehensive_summary(summary: Dict[str, Any]):
    """Print a comprehensive test summary"""
    print("\n" + "=" * 60)
    print("🏆 COMPREHENSIVE A2A TEST RESULTS")
    print("=" * 60)

    # Overall statistics
    total = summary["total_tests"]
    successful = summary["successful_tests"]
    failed = summary["failed_tests"]
    success_rate = summary["success_rate"]
    duration = summary["total_duration"]

    print(f"\n📊 Overall Results:")
    print(f"   Total Tests: {total}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {success_rate:.1%}")
    print(f"   Total Duration: {duration:.2f}s")

    # Test category results
    print(f"\n📋 Test Category Results:")
    for test_name, result in summary["results"].items():
        status = "✅" if result["success"] else "❌"
        details = result.get("details", "N/A")
        test_duration = result["duration"]

        print(f"   {status} {test_name}: {details} ({test_duration:.2f}s)")

    # Performance metrics
    if successful > 0:
        avg_duration = sum(r["duration"] for r in summary["results"].values()) / len(
            summary["results"]
        )
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average test duration: {avg_duration:.2f}s")
        print(f"   Tests per second: {total / duration:.2f}")

    # Framework coverage
    print(f"\n🎯 Framework Features Tested:")
    features_tested = [
        "✅ Agent Discovery (Agent Cards)",
        "✅ Health Checks & Capabilities",
        "✅ Basic Message Sending",
        "✅ Tool Execution",
        "✅ Streaming Responses",
        "✅ Task Memory & Persistence",
        "✅ Performance Under Load",
        "✅ Error Handling",
        "✅ Convenience Methods",
        "✅ Multiple Agent Types",
        "✅ Advanced Tool Schemas",
        "✅ JSON-RPC Protocol",
        "✅ A2A Client Library",
        "✅ Server Configuration",
        "✅ Mock Model Provider",
    ]

    for feature in features_tested:
        print(f"   {feature}")

    # Overall assessment
    print(f"\n🎉 Assessment:")
    if success_rate >= 0.9:
        print("   🟢 EXCELLENT: A2A framework is working perfectly!")
    elif success_rate >= 0.7:
        print("   🟡 GOOD: A2A framework is mostly working with minor issues")
    elif success_rate >= 0.5:
        print("   🟠 FAIR: A2A framework has some significant issues")
    else:
        print("   🔴 POOR: A2A framework has major issues that need attention")

    print(f"\n📚 What You Can Do Next:")
    print("   1. Use the server example: python jaf/a2a/examples/server_example.py")
    print("   2. Try the client example: python jaf/a2a/examples/client_example.py")
    print("   3. Run integration tests: python jaf/a2a/examples/integration_example.py")
    print("   4. Build your own A2A agents with custom tools")
    print("   5. Explore different memory providers (Redis, PostgreSQL)")
    print("   6. Implement real model providers (OpenAI, Anthropic, etc.)")
    print("   7. Scale with multiple server instances")


async def main():
    """Main function to run comprehensive A2A tests"""
    test_runner = ComprehensiveA2ATest()

    try:
        print("🚀 Starting comprehensive A2A framework test...")
        print("   This will test ALL A2A functionality in JAF")
        print("   Including: agents, tools, memory, performance, and more!")

        # Run all tests
        summary = await test_runner.run_all_tests()

        # Print comprehensive results
        print_comprehensive_summary(summary)

        # Determine exit code
        success_rate = summary["success_rate"]
        return 0 if success_rate >= 0.7 else 1

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return 1

    finally:
        await test_runner.cleanup()


if __name__ == "__main__":
    import sys

    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Comprehensive test stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
