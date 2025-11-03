"""
JAF Visualization - Example Usage

Example demonstrating how to use the Graphviz visualization functionality.
This script creates sample agents and tools, then generates various visualizations
to showcase the capabilities of the visualization system.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from ..core.types import Agent, ModelConfig, RunState, ToolSchema
from .graphviz import generate_agent_graph, generate_runner_graph, generate_tool_graph
from .types import GraphOptions

# ========== Example Tool Implementations ==========


@dataclass
class CalculateArgs:
    """Arguments for calculator tool."""

    expression: str


class ExampleCalculatorTool:
    """Example calculator tool for mathematical operations."""

    @property
    def schema(self) -> ToolSchema[CalculateArgs]:
        """Tool schema including name, description, and parameter validation."""
        return ToolSchema(
            name="calculator",
            description="Performs basic arithmetic operations",
            parameters=CalculateArgs,
        )

    async def execute(self, args: CalculateArgs, context: Any) -> str:
        """Execute the calculator tool."""
        try:
            # Simple calculator (in real implementation, use safer evaluation)
            result = eval(args.expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: Invalid mathematical expression - {e!s}"


@dataclass
class WeatherArgs:
    """Arguments for weather tool."""

    location: str


class ExampleWeatherTool:
    """Example weather tool for getting weather information."""

    @property
    def schema(self) -> ToolSchema[WeatherArgs]:
        """Tool schema including name, description, and parameter validation."""
        return ToolSchema(
            name="weather",
            description="Gets current weather for a location",
            parameters=WeatherArgs,
        )

    async def execute(self, args: WeatherArgs, context: Any) -> str:
        """Execute the weather tool."""
        # Mock weather data
        return f"Current weather in {args.location}: 72°F, sunny"


@dataclass
class SearchArgs:
    """Arguments for search tool."""

    query: str


class ExampleSearchTool:
    """Example search tool for web searches."""

    @property
    def schema(self) -> ToolSchema[SearchArgs]:
        """Tool schema including name, description, and parameter validation."""
        return ToolSchema(
            name="search", description="Searches the web for information", parameters=SearchArgs
        )

    async def execute(self, args: SearchArgs, context: Any) -> str:
        """Execute the search tool."""
        return f"Search results for: {args.query}"


# ========== Example Agent Definitions ==========


def create_example_agent() -> Agent:
    """Create a multi-purpose example assistant agent."""
    calculator_tool = ExampleCalculatorTool()
    weather_tool = ExampleWeatherTool()
    search_tool = ExampleSearchTool()

    def instructions(state: RunState[Any]) -> str:
        return (
            "I am a helpful assistant that can perform calculations, "
            "check weather, and search for information."
        )

    return Agent(
        name="ExampleAssistant",
        instructions=instructions,
        tools=[calculator_tool, weather_tool, search_tool],
        model_config=ModelConfig(name="gpt-4"),
        handoffs=["MathSpecialist", "WeatherBot"],
    )


def create_math_specialist() -> Agent:
    """Create a specialized math agent."""
    calculator_tool = ExampleCalculatorTool()

    def instructions(state: RunState[Any]) -> str:
        return "I specialize in mathematical calculations and problem solving."

    return Agent(
        name="MathSpecialist",
        instructions=instructions,
        tools=[calculator_tool],
        model_config=ModelConfig(name="gpt-3.5-turbo"),
    )


def create_weather_bot() -> Agent:
    """Create a specialized weather agent."""
    weather_tool = ExampleWeatherTool()

    def instructions(state: RunState[Any]) -> str:
        return "I provide weather information and forecasts."

    return Agent(
        name="WeatherBot",
        instructions=instructions,
        tools=[weather_tool],
        model_config=ModelConfig(name="gpt-3.5-turbo"),
    )


def create_search_specialist() -> Agent:
    """Create a specialized search agent."""
    search_tool = ExampleSearchTool()

    def instructions(state: RunState[Any]) -> str:
        return "I specialize in finding and retrieving information from various sources."

    return Agent(
        name="SearchSpecialist",
        instructions=instructions,
        tools=[search_tool],
        model_config=ModelConfig(name="gpt-4"),
    )


# ========== Example Functions ==========


async def run_visualization_examples() -> None:
    """Run comprehensive visualization examples."""
    print("🎨 Running JAF Visualization Examples...\n")

    # Ensure output directory exists
    os.makedirs("./examples", exist_ok=True)

    try:
        # Create example agents
        example_agent = create_example_agent()
        math_specialist = create_math_specialist()
        weather_bot = create_weather_bot()
        search_specialist = create_search_specialist()

        agents = [example_agent, math_specialist, weather_bot, search_specialist]

        # 1. Generate Agent Graph
        print("📊 Generating agent visualization...")
        agent_result = await generate_agent_graph(
            agents,
            GraphOptions(
                title="JAF Agent System",
                output_path="./examples/agent-graph.png",
                output_format="png",
                show_tool_details=True,
                show_sub_agents=True,
                color_scheme="modern",
            ),
        )

        if agent_result.success:
            print(f"✅ Agent graph generated: {agent_result.output_path}")
        else:
            print(f"❌ Agent graph failed: {agent_result.error}")

        # 2. Generate Tool Graph
        print("\n🔧 Generating tool visualization...")
        all_tools = [ExampleCalculatorTool(), ExampleWeatherTool(), ExampleSearchTool()]

        tool_result = await generate_tool_graph(
            all_tools,
            GraphOptions(
                title="JAF Tool Ecosystem",
                output_path="./examples/tool-graph.png",
                output_format="png",
                layout="circo",
                color_scheme="default",
            ),
        )

        if tool_result.success:
            print(f"✅ Tool graph generated: {tool_result.output_path}")
        else:
            print(f"❌ Tool graph failed: {tool_result.error}")

        # 3. Generate Runner Visualization
        print("\n🏃 Generating runner visualization...")
        agent_registry = {agent.name: agent for agent in agents}

        runner_result = await generate_runner_graph(
            agent_registry,
            GraphOptions(
                title="JAF Runner Architecture",
                output_path="./examples/runner-architecture.png",
                output_format="png",
                color_scheme="modern",
            ),
        )

        if runner_result.success:
            print(f"✅ Runner graph generated: {runner_result.output_path}")
        else:
            print(f"❌ Runner graph failed: {runner_result.error}")

        # 4. Generate different color schemes
        print("\n🎨 Generating alternative color schemes...")

        for scheme in ["default", "modern", "minimal"]:
            scheme_result = await generate_agent_graph(
                [example_agent],
                GraphOptions(
                    title=f"JAF Agent ({scheme} theme)",
                    output_path=f"./examples/agent-{scheme}.png",
                    output_format="png",
                    color_scheme=scheme,
                    show_tool_details=True,
                ),
            )

            if scheme_result.success:
                print(f"✅ {scheme} theme generated: {scheme_result.output_path}")
            else:
                print(f"❌ {scheme} theme failed: {scheme_result.error}")

        # 5. Generate different formats
        print("\n📄 Generating different output formats...")

        for fmt in ["png", "svg", "pdf"]:
            format_result = await generate_agent_graph(
                [example_agent, math_specialist],
                GraphOptions(
                    title="JAF Multi-Format Example",
                    output_path=f"./examples/multi-format.{fmt}",
                    output_format=fmt,
                    color_scheme="modern",
                ),
            )

            if format_result.success:
                print(f"✅ {fmt.upper()} format generated: {format_result.output_path}")
            else:
                print(f"❌ {fmt.upper()} format failed: {format_result.error}")

        print("\n🎉 All visualization examples completed!")
        print("\n📁 Generated files:")
        print("   - ./examples/agent-graph.png")
        print("   - ./examples/tool-graph.png")
        print("   - ./examples/runner-architecture.png")
        print("   - ./examples/agent-default.png")
        print("   - ./examples/agent-modern.png")
        print("   - ./examples/agent-minimal.png")
        print("   - ./examples/multi-format.png")
        print("   - ./examples/multi-format.svg")
        print("   - ./examples/multi-format.pdf")

    except Exception as error:
        print(f"❌ Error running visualization examples: {error}")


async def quick_start_visualization(agent: Agent, output_path: str = None) -> None:
    """
    Quick start function for visualizing a single agent.

    Args:
        agent: Agent to visualize
        output_path: Optional custom output path
    """
    print(f"🚀 Quick visualization for agent: {agent.name}")

    result = await generate_agent_graph(
        [agent],
        GraphOptions(
            output_path=output_path or f"./agent-{agent.name.lower()}.png",
            output_format="png",
            color_scheme="modern",
            show_tool_details=True,
            show_sub_agents=True,
        ),
    )

    if result.success:
        print(f"✅ Visualization saved to: {result.output_path}")
    else:
        print(f"❌ Visualization failed: {result.error}")


async def demo_basic_usage() -> None:
    """Demonstrate basic usage patterns."""
    print("📚 JAF Visualization - Basic Usage Demo\n")

    # Create a simple agent
    calculator_tool = ExampleCalculatorTool()

    def simple_instructions(state: RunState[Any]) -> str:
        return "I am a simple calculator agent."

    simple_agent = Agent(
        name="SimpleCalculator",
        instructions=simple_instructions,
        tools=[calculator_tool],
        model_config=ModelConfig(name="gpt-3.5-turbo"),
    )

    # Generate basic visualization
    result = await generate_agent_graph(
        [simple_agent],
        GraphOptions(title="Simple Calculator Agent", output_path="./simple-agent.png"),
    )

    if result.success:
        print(f"✅ Basic visualization created: {result.output_path}")
        print("📄 DOT source preview:")
        print(result.graph_dot[:200] + "..." if result.graph_dot else "No DOT source available")
    else:
        print(f"❌ Basic visualization failed: {result.error}")


# ========== CLI Integration ==========


async def main() -> None:
    """Main function for CLI execution."""
    print("🎨 JAF Visualization Examples\n")

    # Check if graphviz is available
    try:
        from graphviz import Digraph

        test_graph = Digraph()
        print("✅ Graphviz Python package available")
    except ImportError:
        print("❌ Graphviz Python package not installed")
        print('   Install with: pip install "jaf-py[visualization]"')
        return

    try:
        # Test basic Graphviz system dependency
        test_graph = Digraph()
        test_graph.node("test", "Test Node")
        test_graph.source  # This will work if graphviz is available
        print("✅ Graphviz system dependency available")
    except Exception:
        print("❌ Graphviz system dependency not found")
        print("   Install with: brew install graphviz (macOS) or apt-get install graphviz (Ubuntu)")
        return

    # Run examples
    await demo_basic_usage()
    print()
    await run_visualization_examples()


if __name__ == "__main__":
    asyncio.run(main())
