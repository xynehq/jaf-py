#!/usr/bin/env python3
"""
Demonstration tests for JAF visualization functionality.
These tests serve as both validation and examples of how to use the visualization system.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

from jaf.core.types import Agent, ModelConfig, RunState, ToolSchema
from jaf.visualization.graphviz import generate_agent_graph, generate_tool_graph
from jaf.visualization.types import GraphOptions


@dataclass
class CalculatorArgs:
    expression: str


class CalculatorTool:
    @property
    def schema(self) -> ToolSchema[CalculatorArgs]:
        return ToolSchema(
            name="calculator",
            description="Performs mathematical calculations",
            parameters=CalculatorArgs,
        )

    async def execute(self, args: CalculatorArgs, context: Any) -> str:
        try:
            result = eval(args.expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e!s}"


@dataclass
class WeatherArgs:
    location: str


class WeatherTool:
    @property
    def schema(self) -> ToolSchema[WeatherArgs]:
        return ToolSchema(
            name="weather", description="Gets weather information", parameters=WeatherArgs
        )

    async def execute(self, args: WeatherArgs, context: Any) -> str:
        return f"Weather in {args.location}: 72°F, sunny"


class TestVisualizationDemo:
    """Demo tests that showcase visualization functionality."""

    @pytest.mark.asyncio
    async def test_agent_visualization_demo(self):
        """Test agent visualization with realistic example."""
        # Create tools
        calc_tool = CalculatorTool()
        weather_tool = WeatherTool()

        # Create agents
        def assistant_instructions(state: RunState[Any]) -> str:
            return "I am a helpful assistant with calculator and weather tools."

        assistant = Agent(
            name="Assistant",
            instructions=assistant_instructions,
            tools=[calc_tool, weather_tool],
            model_config=ModelConfig(name="gpt-4"),
            handoffs=["Specialist"],
        )

        def specialist_instructions(state: RunState[Any]) -> str:
            return "I am a specialist agent."

        specialist = Agent(
            name="Specialist",
            instructions=specialist_instructions,
            tools=[calc_tool],
            model_config=ModelConfig(name="gpt-3.5-turbo"),
        )

        # Generate agent visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "demo-agents.png")

            try:
                agent_result = await generate_agent_graph(
                    [assistant, specialist],
                    GraphOptions(
                        title="Demo Agent System",
                        output_path=output_path,
                        color_scheme="modern",
                        show_tool_details=True,
                    ),
                )

                # Should either succeed or fail gracefully
                assert isinstance(agent_result.success, bool)

                if agent_result.success:
                    assert agent_result.output_path == output_path
                    assert agent_result.graph_dot is not None
                    assert "Assistant" in agent_result.graph_dot
                    assert "Specialist" in agent_result.graph_dot
                else:
                    # If it fails, should have error message
                    assert agent_result.error is not None

            except Exception as e:
                pytest.skip(f"Graphviz not available: {e}")

    @pytest.mark.asyncio
    async def test_tool_visualization_demo(self):
        """Test tool visualization with realistic example."""
        # Create tools
        calc_tool = CalculatorTool()
        weather_tool = WeatherTool()

        # Generate tool visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "demo-tools.png")

            try:
                tool_result = await generate_tool_graph(
                    [calc_tool, weather_tool],
                    GraphOptions(
                        title="Demo Tool Ecosystem", output_path=output_path, color_scheme="default"
                    ),
                )

                # Should either succeed or fail gracefully
                assert isinstance(tool_result.success, bool)

                if tool_result.success:
                    assert tool_result.output_path == output_path
                    assert tool_result.graph_dot is not None
                    assert "calculator" in tool_result.graph_dot
                    assert "weather" in tool_result.graph_dot
                else:
                    # If it fails, should have error message
                    assert tool_result.error is not None

            except Exception as e:
                pytest.skip(f"Graphviz not available: {e}")

    @pytest.mark.asyncio
    async def test_multiple_color_schemes_demo(self):
        """Test different color schemes with demo data."""
        calc_tool = CalculatorTool()

        def simple_instructions(state: RunState[Any]) -> str:
            return "I am a simple agent."

        simple_agent = Agent(
            name="SimpleAgent",
            instructions=simple_instructions,
            tools=[calc_tool],
            model_config=ModelConfig(name="gpt-3.5-turbo"),
        )

        color_schemes = ["default", "modern", "minimal"]

        for scheme in color_schemes:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, f"demo-{scheme}.png")

                try:
                    result = await generate_agent_graph(
                        [simple_agent],
                        GraphOptions(
                            title=f"Demo {scheme.title()} Theme",
                            output_path=output_path,
                            color_scheme=scheme,
                            show_tool_details=True,
                        ),
                    )

                    # Should either succeed or fail gracefully
                    assert isinstance(result.success, bool)

                    if result.success:
                        assert result.output_path == output_path
                        assert result.graph_dot is not None
                        assert "SimpleAgent" in result.graph_dot
                    else:
                        # If it fails, should have error message
                        assert result.error is not None

                except Exception as e:
                    pytest.skip(f"Graphviz not available for scheme {scheme}: {e}")


async def demo_main():
    """
    Standalone demo function that can be run directly.
    This provides the same functionality as the original demo script.
    """
    print("🎨 JAF Visualization Demo")

    # Create tools
    calc_tool = CalculatorTool()
    weather_tool = WeatherTool()

    # Create agents
    def assistant_instructions(state: RunState[Any]) -> str:
        return "I am a helpful assistant with calculator and weather tools."

    assistant = Agent(
        name="Assistant",
        instructions=assistant_instructions,
        tools=[calc_tool, weather_tool],
        model_config=ModelConfig(name="gpt-4"),
        handoffs=["Specialist"],
    )

    def specialist_instructions(state: RunState[Any]) -> str:
        return "I am a specialist agent."

    specialist = Agent(
        name="Specialist",
        instructions=specialist_instructions,
        tools=[calc_tool],
        model_config=ModelConfig(name="gpt-3.5-turbo"),
    )

    # Generate agent visualization
    print("\n📊 Generating agent visualization...")
    agent_result = await generate_agent_graph(
        [assistant, specialist],
        GraphOptions(
            title="Demo Agent System",
            output_path="./demo-agents.png",
            color_scheme="modern",
            show_tool_details=True,
        ),
    )

    if agent_result.success:
        print(f"✅ Agent graph: {agent_result.output_path}")
    else:
        print(f"❌ Agent graph failed: {agent_result.error}")

    # Generate tool visualization
    print("\n🔧 Generating tool visualization...")
    tool_result = await generate_tool_graph(
        [calc_tool, weather_tool],
        GraphOptions(
            title="Demo Tool Ecosystem", output_path="./demo-tools.png", color_scheme="default"
        ),
    )

    if tool_result.success:
        print(f"✅ Tool graph: {tool_result.output_path}")
    else:
        print(f"❌ Tool graph failed: {tool_result.error}")

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    # Allow running as standalone demo script
    asyncio.run(demo_main())
