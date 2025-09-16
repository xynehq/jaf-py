"""
Advanced Parallel Agents Demo for JAF Framework

This example demonstrates the new parallel agent execution feature that allows
you to configure which agents run in parallel groups with different execution
strategies and result aggregation methods.

New Features Demonstrated:
1. ParallelAgentGroup - Group agents for parallel execution
2. ParallelExecutionConfig - Configure execution behavior
3. Multiple result aggregation strategies
4. Inter-group execution modes (sequential vs parallel)
5. Timeout handling for parallel operations
6. Convenient helper functions for common patterns

Usage:
    python examples/advanced_parallel_agents_demo.py
"""

import asyncio
import os
import logging
from dataclasses import dataclass
from typing import Dict

# JAF Imports
from jaf import Agent, make_litellm_provider
from jaf.core.types import RunState, RunConfig, Message, generate_run_id, generate_trace_id, ContentRole
from jaf.core.engine import run
from jaf.core.parallel_agents import (
    ParallelAgentGroup,
    ParallelExecutionConfig,
    create_parallel_agents_tool,
    create_simple_parallel_tool,
    create_language_specialists_tool,
    create_domain_experts_tool
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_litellm_provider():
    """Setup LiteLLM provider using environment variables."""
    base_url = os.getenv('LITELLM_BASE_URL', 'https://grid.ai.juspay.net/')
    api_key = os.getenv('LITELLM_API_KEY')
    
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is required")
    
    return make_litellm_provider(base_url, api_key)

@dataclass
class DemoContext:
    """Context for the demo."""
    user_id: str = "demo_user"
    session_id: str = "parallel_demo_001"
    experiment_type: str = "advanced_parallel"

def create_specialist_agents() -> Dict[str, Agent]:
    """Create various specialist agents for parallel execution."""
    
    # Language specialists
    spanish_agent = Agent(
        name='spanish_specialist',
        instructions=lambda state: '''Eres un especialista en español. 
        Responde SIEMPRE en español. Traduce y explica conceptos en español.
        Sé amigable y usa expresiones auténticas españolas.''',
        tools=[]
    )
    
    french_agent = Agent(
        name='french_specialist', 
        instructions=lambda state: '''Tu es un spécialiste français.
        Réponds TOUJOURS en français. Traduis et explique en français.
        Sois amical et utilise des expressions françaises authentiques.''',
        tools=[]
    )
    
    german_agent = Agent(
        name='german_specialist',
        instructions=lambda state: '''Du bist ein deutscher Spezialist.
        Antworte IMMER auf Deutsch. Übersetze und erkläre auf Deutsch.
        Sei freundlich und verwende authentische deutsche Ausdrücke.''',
        tools=[]
    )
    
    # Domain experts
    tech_expert = Agent(
        name='tech_expert',
        instructions=lambda state: '''You are a technology expert specializing in software architecture,
        programming languages, and system design. Provide technical insights and recommendations.
        Focus on practical solutions and best practices.''',
        tools=[]
    )
    
    business_expert = Agent(
        name='business_expert', 
        instructions=lambda state: '''You are a business strategy expert specializing in market analysis,
        business models, and strategic planning. Provide business insights and recommendations.
        Focus on practical business solutions and market opportunities.''',
        tools=[]
    )
    
    creative_expert = Agent(
        name='creative_expert',
        instructions=lambda state: '''You are a creative expert specializing in design thinking,
        innovation, and creative problem-solving. Provide creative insights and innovative solutions.
        Think outside the box and suggest novel approaches.''',
        tools=[]
    )
    
    return {
        'spanish': spanish_agent,
        'french': french_agent,
        'german': german_agent,
        'tech': tech_expert,
        'business': business_expert,
        'creative': creative_expert
    }

def create_orchestrator_with_parallel_tools(specialists: Dict[str, Agent]) -> Agent:
    """Create an orchestrator that uses parallel agent tools."""
    
    # Create language specialists tool
    language_agents = {k: v for k, v in specialists.items() if k in ['spanish', 'french', 'german']}
    language_tool = create_language_specialists_tool(
        language_agents,
        tool_name="consult_language_specialists",
        timeout=30.0
    )
    
    # Create domain experts tool
    expert_agents = {k: v for k, v in specialists.items() if k in ['tech', 'business', 'creative']}
    experts_tool = create_domain_experts_tool(
        expert_agents,
        tool_name="consult_domain_experts",
        result_aggregation="combine",
        timeout=45.0
    )
    
    # Create a custom parallel configuration for mixed groups
    mixed_groups = [
        ParallelAgentGroup(
            name="rapid_response_team",
            agents=[specialists['tech'], specialists['creative']],
            shared_input=True,
            result_aggregation="combine",
            timeout=20.0,
            metadata={"priority": "high", "type": "rapid_response"}
        ),
        ParallelAgentGroup(
            name="analysis_team", 
            agents=[specialists['business'], specialists['spanish']],
            shared_input=True,
            result_aggregation="first",
            timeout=25.0,
            metadata={"priority": "medium", "type": "analysis"}
        )
    ]
    
    mixed_parallel_tool = create_parallel_agents_tool(
        mixed_groups,
        tool_name="consult_mixed_teams",
        tool_description="Consult rapid response and analysis teams in parallel",
        inter_group_execution="parallel",  # Execute groups in parallel
        global_timeout=60.0
    )
    
    # Create simple parallel tool for quick consultation
    quick_consult_tool = create_simple_parallel_tool(
        agents=[specialists['tech'], specialists['business']],
        group_name="quick_consult",
        tool_name="quick_parallel_consult",
        shared_input=True,
        result_aggregation="combine",
        timeout=15.0
    )
    
    def orchestrator_instructions(state):
        return '''You are an advanced orchestrator that can coordinate multiple specialist teams in parallel.

Available parallel execution tools:
1. consult_language_specialists - Get responses from Spanish, French, and German specialists in parallel
2. consult_domain_experts - Get insights from tech, business, and creative experts in parallel  
3. consult_mixed_teams - Coordinate rapid response (tech+creative) and analysis (business+spanish) teams
4. quick_parallel_consult - Quick consultation with tech and business experts

Use these tools based on the user's needs:
- For multilingual responses: use consult_language_specialists
- For comprehensive expert analysis: use consult_domain_experts
- For complex problems requiring multiple perspectives: use consult_mixed_teams
- For quick technical/business insights: use quick_parallel_consult

You can also call multiple tools in the same response to get even more parallel execution.
Explain the results from parallel executions and synthesize insights.'''
    
    return Agent(
        name='advanced_orchestrator',
        instructions=orchestrator_instructions,
        tools=[language_tool, experts_tool, mixed_parallel_tool, quick_consult_tool]
    )

async def demo_simple_parallel():
    """Demo 1: Simple parallel execution."""
    print("\n🚀 Demo 1: Simple Parallel Execution")
    print("=" * 50)
    
    try:
        model_provider = setup_litellm_provider()
        specialists = create_specialist_agents()
        
        # Create simple parallel tool
        parallel_tool = create_simple_parallel_tool(
            agents=[specialists['spanish'], specialists['french']],
            group_name="translators",
            tool_name="translate_parallel", 
            shared_input=True,
            result_aggregation="combine",
            timeout=30.0
        )
        
        # Create simple orchestrator
        orchestrator = Agent(
            name='simple_orchestrator',
            instructions=lambda state: 'Use the translate_parallel tool to get translations in multiple languages.',
            tools=[parallel_tool]
        )
        
        # Run
        context = DemoContext(experiment_type="simple_parallel")
        
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(
                role=ContentRole.USER,
                content="Please translate 'Hello, how are you today?' into multiple languages"
            )],
            current_agent_name='simple_orchestrator',
            context=context,
            turn_count=0
        )
        
        config = RunConfig(
            agent_registry={'simple_orchestrator': orchestrator},
            model_provider=model_provider,
            max_turns=3
        )
        
        result = await run(initial_state, config)
        
        print("✅ Simple parallel execution completed")
        print(f"Status: {result.outcome.status}")
        if hasattr(result.outcome, 'output'):
            print(f"Output: {result.outcome.output}")
            
    except Exception as e:
        print(f"❌ Simple parallel demo failed: {e}")

async def demo_advanced_parallel():
    """Demo 2: Advanced parallel execution with multiple groups."""
    print("\n🚀 Demo 2: Advanced Parallel Execution")
    print("=" * 50)
    
    try:
        model_provider = setup_litellm_provider()
        specialists = create_specialist_agents()
        orchestrator = create_orchestrator_with_parallel_tools(specialists)
        
        # Create comprehensive agent registry
        agent_registry = {'advanced_orchestrator': orchestrator}
        agent_registry.update(specialists)
        
        context = DemoContext(experiment_type="advanced_parallel")
        
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(
                role=ContentRole.USER,
                content="I need help developing a multilingual AI chatbot for customer service. I want technical architecture advice, business strategy insights, and creative UX ideas. Please consult your specialist teams in parallel."
            )],
            current_agent_name='advanced_orchestrator',
            context=context,
            turn_count=0
        )
        
        config = RunConfig(
            agent_registry=agent_registry,
            model_provider=model_provider,
            max_turns=5,
            model_override='gemini-2.5-pro'
        )
        
        result = await run(initial_state, config)
        
        print("✅ Advanced parallel execution completed")
        print(f"Status: {result.outcome.status}")
        print(f"Total turns: {result.final_state.turn_count}")
        
        # Show conversation flow
        print("\n📝 Conversation Summary:")
        for i, msg in enumerate(result.final_state.messages[-3:], 1):  # Show last 3 messages
            role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}.get(msg.role, "❓")
            content_preview = (msg.content[:100] + "...") if len(msg.content) > 100 else msg.content
            print(f"  {i}. {role_emoji} {msg.role.upper()}: {content_preview}")
            
    except Exception as e:
        print(f"❌ Advanced parallel demo failed: {e}")
        import traceback
        traceback.print_exc()

async def demo_custom_aggregation():
    """Demo 3: Custom result aggregation."""
    print("\n🚀 Demo 3: Custom Result Aggregation")
    print("=" * 50)
    
    try:
        model_provider = setup_litellm_provider()
        specialists = create_specialist_agents()
        
        # Custom aggregation function
        def consensus_aggregator(results):
            """Custom aggregator that looks for consensus."""
            if len(results) < 2:
                return {"consensus": False, "single_result": results[0] if results else "No results"}
            
            # Simple consensus detection (could be more sophisticated)
            common_themes = []
            for result in results:
                if "recommend" in result.lower():
                    common_themes.append("recommendation")
                if "suggest" in result.lower():
                    common_themes.append("suggestion")
            
            return {
                "consensus": len(common_themes) >= len(results) // 2,
                "common_themes": list(set(common_themes)),
                "all_results": results,
                "summary": f"Consensus reached on {len(set(common_themes))} themes" if common_themes else "No clear consensus"
            }
        
        # Create group with custom aggregation
        custom_group = ParallelAgentGroup(
            name="consensus_team",
            agents=[specialists['tech'], specialists['business'], specialists['creative']],
            shared_input=True,
            result_aggregation="custom",
            custom_aggregator=consensus_aggregator,
            timeout=30.0
        )
        
        custom_tool = create_parallel_agents_tool(
            [custom_group],
            tool_name="get_team_consensus",
            tool_description="Get consensus from multiple experts"
        )
        
        orchestrator = Agent(
            name='consensus_orchestrator',
            instructions=lambda state: 'Use the get_team_consensus tool to gather expert opinions and find consensus.',
            tools=[custom_tool]
        )
        
        context = DemoContext(experiment_type="custom_aggregation")
        
        initial_state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=[Message(
                role=ContentRole.USER,
                content="What technology stack would you recommend for building a scalable e-commerce platform?"
            )],
            current_agent_name='consensus_orchestrator',
            context=context,
            turn_count=0
        )
        
        config = RunConfig(
            agent_registry={'consensus_orchestrator': orchestrator},
            model_provider=model_provider,
            max_turns=3
        )
        
        result = await run(initial_state, config)
        
        print("✅ Custom aggregation demo completed")
        print(f"Status: {result.outcome.status}")
        if hasattr(result.outcome, 'output'):
            print(f"Output: {result.outcome.output}")
            
    except Exception as e:
        print(f"❌ Custom aggregation demo failed: {e}")

async def run_all_demos():
    """Run all parallel agent demos."""
    print("🤖 JAF Advanced Parallel Agents Demo")
    print("Demonstrating new parallel execution capabilities")
    print("=" * 60)
    
    await demo_simple_parallel()
    await demo_advanced_parallel() 
    await demo_custom_aggregation()
    
    print("\n🎉 All parallel agent demos completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Simple parallel agent execution")
    print("✅ Advanced multi-group parallel execution")
    print("✅ Multiple result aggregation strategies")
    print("✅ Custom aggregation functions")
    print("✅ Timeout handling for parallel operations")
    print("✅ Inter-group execution modes")
    print("✅ Language specialists and domain experts patterns")

if __name__ == "__main__":
    asyncio.run(run_all_demos())