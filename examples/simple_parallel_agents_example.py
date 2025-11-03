"""
Simple Parallel Agents Example

This example shows the easiest way to use the new parallel agent execution feature.
It demonstrates how to quickly set up agents to run in parallel with minimal configuration.

Usage:
    python examples/simple_parallel_agents_example.py
"""

import asyncio
from jaf import Agent
from jaf.core.parallel_agents import create_simple_parallel_tool


# Example 1: Simple parallel execution
def example_simple_parallel():
    """Create agents and run them in parallel with one line of code."""

    # Create some simple agents
    math_agent = Agent(
        name="math_expert",
        instructions=lambda state: "You are a math expert. Solve mathematical problems and explain your reasoning.",
        tools=[],
    )

    science_agent = Agent(
        name="science_expert",
        instructions=lambda state: "You are a science expert. Explain scientific concepts and phenomena.",
        tools=[],
    )

    history_agent = Agent(
        name="history_expert",
        instructions=lambda state: "You are a history expert. Provide historical context and facts.",
        tools=[],
    )

    # Create a parallel tool - this is all you need!
    parallel_experts_tool = create_simple_parallel_tool(
        agents=[math_agent, science_agent, history_agent],
        group_name="expert_panel",
        tool_name="consult_experts",
        shared_input=True,  # All agents get the same input
        result_aggregation="combine",  # Combine all results
        timeout=300.0,  # 300 second timeout
    )

    print("✅ Created parallel experts tool!")
    print(f"Tool name: {parallel_experts_tool.schema.name}")
    print(f"Tool description: {parallel_experts_tool.schema.description}")
    print(f"Number of agents: {len([math_agent, science_agent, history_agent])}")

    return parallel_experts_tool


# Example 2: Language specialists
def example_language_specialists():
    """Create language specialist agents that run in parallel."""
    from jaf.core.parallel_agents import create_language_specialists_tool

    # Create language agents
    spanish_agent = Agent(
        name="spanish_translator",
        instructions=lambda state: "Translate to Spanish and respond in Spanish only.",
        tools=[],
    )

    french_agent = Agent(
        name="french_translator",
        instructions=lambda state: "Translate to French and respond in French only.",
        tools=[],
    )

    italian_agent = Agent(
        name="italian_translator",
        instructions=lambda state: "Translate to Italian and respond in Italian only.",
        tools=[],
    )

    # Create language specialists tool
    language_tool = create_language_specialists_tool(
        language_agents={
            "spanish": spanish_agent,
            "french": french_agent,
            "italian": italian_agent,
        },
        tool_name="translate_to_multiple_languages",
        timeout=25.0,
    )

    print("✅ Created language specialists tool!")
    print(f"Tool name: {language_tool.schema.name}")
    print("Supported languages: Spanish, French, Italian")

    return language_tool


# Example 3: Domain experts
def example_domain_experts():
    """Create domain expert agents that run in parallel."""
    from jaf.core.parallel_agents import create_domain_experts_tool

    # Create domain expert agents
    tech_agent = Agent(
        name="tech_advisor",
        instructions=lambda state: "Provide technical advice on software, systems, and technology.",
        tools=[],
    )

    business_agent = Agent(
        name="business_advisor",
        instructions=lambda state: "Provide business strategy and market analysis advice.",
        tools=[],
    )

    legal_agent = Agent(
        name="legal_advisor",
        instructions=lambda state: "Provide legal considerations and compliance advice.",
        tools=[],
    )

    # Create domain experts tool
    experts_tool = create_domain_experts_tool(
        expert_agents={"technology": tech_agent, "business": business_agent, "legal": legal_agent},
        tool_name="consult_advisory_board",
        result_aggregation="combine",  # Get all perspectives
        timeout=45.0,
    )

    print("✅ Created domain experts tool!")
    print(f"Tool name: {experts_tool.schema.name}")
    print("Expert domains: Technology, Business, Legal")

    return experts_tool


# Example 4: Using parallel tools in an orchestrator
def example_orchestrator_with_parallel_tools():
    """Create an orchestrator that uses parallel tools."""

    # Get the tools from previous examples
    experts_tool = example_simple_parallel()
    language_tool = example_language_specialists()
    advisory_tool = example_domain_experts()

    # Create orchestrator that uses all parallel tools
    orchestrator = Agent(
        name="super_orchestrator",
        instructions=lambda state: """You are a super orchestrator with access to multiple parallel agent teams:

1. consult_experts - Math, Science, and History experts (for educational questions)
2. translate_to_multiple_languages - Spanish, French, Italian translators (for translation needs)  
3. consult_advisory_board - Tech, Business, Legal advisors (for project advice)

Choose the right parallel tool(s) based on the user's request. You can even use multiple tools in one response for comprehensive help!""",
        tools=[experts_tool, language_tool, advisory_tool],
    )

    print("✅ Created super orchestrator with parallel tools!")
    print(f"Orchestrator: {orchestrator.name}")
    print(f"Available parallel tools: {len(orchestrator.tools)}")

    return orchestrator


def main():
    """Run all examples."""
    print("🚀 Simple Parallel Agents Examples")
    print("=" * 50)

    print("\n1. Simple Parallel Execution:")
    example_simple_parallel()

    print("\n2. Language Specialists:")
    example_language_specialists()

    print("\n3. Domain Experts:")
    example_domain_experts()

    print("\n4. Orchestrator with Parallel Tools:")
    example_orchestrator_with_parallel_tools()

    print("\n🎉 All examples completed!")
    print("\nKey takeaways:")
    print("• Use create_simple_parallel_tool() for basic parallel execution")
    print("• Use create_language_specialists_tool() for multi-language support")
    print("• Use create_domain_experts_tool() for expert consultation")
    print("• Combine multiple parallel tools in one orchestrator")
    print("• JAF automatically handles the parallel execution with asyncio.gather()")


if __name__ == "__main__":
    main()
