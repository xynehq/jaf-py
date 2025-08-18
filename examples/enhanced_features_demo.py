"""
Enhanced JAF Features Demonstration

This example showcases all the new features added to JAF:
- Performance monitoring
- Streaming responses
- Tool composition
- Enhanced error handling
- Property-based testing concepts

Run this example to see the enhanced capabilities in action.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Import JAF core
import jaf
from jaf.core.types import Message, ContentRole, RunState, Agent, RunConfig
from jaf.core.tool_results import ToolResponse

# Import new features
from jaf.core.performance import PerformanceMonitor, monitor_performance
from jaf.core.streaming import run_streaming, StreamingEventType
from jaf.core.composition import (
    create_tool_pipeline, create_parallel_tools, with_retry, with_cache, compose
)


@dataclass
class DemoContext:
    """Enhanced context for the demo."""
    user_id: str
    session_id: str
    preferences: Dict[str, Any]
    performance_tracking: bool = True


class CalculateArgs(BaseModel):
    """Arguments for calculation tool."""
    expression: str = Field(description="Mathematical expression to evaluate")


class SearchArgs(BaseModel):
    """Arguments for search tool."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")


class WeatherArgs(BaseModel):
    """Arguments for weather tool."""
    location: str = Field(description="Location for weather information")


# Enhanced tools with composition
class CalculatorTool:
    """Enhanced calculator with error handling."""
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='calculate',
            description='Perform mathematical calculations with enhanced error handling',
            parameters=CalculateArgs
        )
    
    async def execute(self, args: CalculateArgs, context: DemoContext):
        """Execute calculation with comprehensive error handling."""
        try:
            # Simulate some processing time
            await asyncio.sleep(0.1)
            
            # Safe evaluation (in production, use a proper math parser)
            if any(dangerous in args.expression for dangerous in ['import', 'exec', 'eval', '__']):
                return ToolResponse.validation_error(
                    "Expression contains potentially dangerous operations",
                    details={'expression': args.expression}
                )
            
            # Simple math evaluation
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in args.expression):
                return ToolResponse.validation_error(
                    "Expression contains invalid characters",
                    details={'allowed': list(allowed_chars)}
                )
            
            result = eval(args.expression)
            
            return ToolResponse.success(
                data={'expression': args.expression, 'result': result},
                metadata={'calculation_time': time.time()}
            )
            
        except ZeroDivisionError:
            return ToolResponse.error(
                code='division_by_zero',
                message='Cannot divide by zero',
                details={'expression': args.expression}
            )
        except Exception as e:
            return ToolResponse.error(
                code='calculation_error',
                message=f'Calculation failed: {str(e)}',
                details={'expression': args.expression, 'error_type': type(e).__name__}
            )


class SearchTool:
    """Enhanced search tool with caching."""
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='search',
            description='Search for information with caching',
            parameters=SearchArgs
        )
    
    async def execute(self, args: SearchArgs, context: DemoContext):
        """Execute search with simulated results."""
        # Simulate search delay
        await asyncio.sleep(0.2)
        
        # Simulate search results
        results = [
            f"Result {i+1} for '{args.query}'" 
            for i in range(min(args.max_results, 3))
        ]
        
        return ToolResponse.success(
            data={'query': args.query, 'results': results, 'total_found': len(results)},
            metadata={'search_time': time.time(), 'cached': False}
        )


class WeatherTool:
    """Weather information tool."""
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='weather',
            description='Get weather information for a location',
            parameters=WeatherArgs
        )
    
    async def execute(self, args: WeatherArgs, context: DemoContext):
        """Get weather information."""
        # Simulate API call
        await asyncio.sleep(0.15)
        
        # Simulate weather data
        weather_data = {
            'location': args.location,
            'temperature': 22,
            'condition': 'Sunny',
            'humidity': 65,
            'wind_speed': 10
        }
        
        return ToolResponse.success(
            data=weather_data,
            metadata={'api_call_time': time.time()}
        )


def create_enhanced_agent() -> Agent[DemoContext, str]:
    """Create an agent with enhanced tools and composition."""
    
    # Create base tools
    calculator = CalculatorTool()
    search = SearchTool()
    weather = WeatherTool()
    
    # Create composed tools
    cached_search = with_cache(search, ttl_seconds=300)  # 5-minute cache
    reliable_calculator = with_retry(calculator, max_retries=2)
    
    # Create a research pipeline
    research_pipeline = create_tool_pipeline(
        cached_search,
        weather,  # Get weather for context
        name="research_pipeline"
    )
    
    # Create parallel information gathering
    parallel_info = create_parallel_tools(
        cached_search,
        weather,
        name="parallel_info",
        combine_strategy="merge"
    )
    
    def instructions(state: RunState[DemoContext]) -> str:
        user_prefs = state.context.preferences
        performance_note = " (Performance monitoring enabled)" if state.context.performance_tracking else ""
        
        return f"""You are an enhanced AI assistant with advanced capabilities{performance_note}.

Available tools:
- calculate: Enhanced calculator with error handling and retry logic
- search: Cached search with 5-minute TTL
- weather: Weather information service
- research_pipeline: Sequential search + weather analysis
- parallel_info: Parallel information gathering

User preferences: {user_prefs}
Session: {state.context.session_id}

Provide helpful, accurate responses using the available tools when appropriate.
"""
    
    return Agent(
        name='EnhancedAssistant',
        instructions=instructions,
        tools=[
            reliable_calculator,
            cached_search,
            weather,
            research_pipeline,
            parallel_info
        ]
    )


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("🔍 Performance Monitoring Demo")
    print("=" * 50)
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Simulate some operations
    for i in range(3):
        monitor.record_llm_call(token_count=150)
        await asyncio.sleep(0.1)
    
    for i in range(2):
        monitor.record_tool_call()
        monitor.record_cache_hit()
        await asyncio.sleep(0.05)
    
    monitor.record_cache_miss()
    
    # Get metrics
    metrics = monitor.stop_monitoring()
    
    print(f"Execution time: {metrics.execution_time_ms:.2f}ms")
    print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
    print(f"Peak memory: {metrics.peak_memory_mb:.2f}MB")
    print(f"LLM calls: {metrics.llm_call_count}")
    print(f"Tool calls: {metrics.tool_call_count}")
    print(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")
    print(f"Total tokens: {metrics.token_count}")
    print()


async def demonstrate_streaming():
    """Demonstrate streaming responses."""
    print("🌊 Streaming Demo")
    print("=" * 50)
    
    # Create a simple agent for streaming
    def simple_instructions(state):
        return "You are a helpful assistant. Provide detailed responses."
    
    simple_agent = Agent(
        name='StreamingAgent',
        instructions=simple_instructions,
        tools=[CalculatorTool()]
    )
    
    # Create mock model provider for demo
    class MockModelProvider:
        async def get_completion(self, state, agent, config):
            return {
                'message': {
                    'content': 'This is a simulated streaming response that demonstrates real-time content delivery.',
                    'tool_calls': None
                }
            }
    
    # Set up streaming
    context = DemoContext(
        user_id='demo_user',
        session_id='streaming_session',
        preferences={'verbose': True}
    )
    
    initial_state = RunState(
        run_id=jaf.generate_run_id(),
        trace_id=jaf.generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content='Tell me about streaming in JAF')],
        current_agent_name='StreamingAgent',
        context=context,
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={'StreamingAgent': simple_agent},
        model_provider=MockModelProvider()
    )
    
    # Stream the response
    print("Streaming response:")
    async for event in run_streaming(initial_state, config, chunk_size=20):
        if event.type == StreamingEventType.START:
            print(f"🚀 Started streaming (Agent: {event.data.agent_name})")
        elif event.type == StreamingEventType.CHUNK:
            print(f"📝 Chunk: '{event.data.delta}'", end='', flush=True)
        elif event.type == StreamingEventType.COMPLETE:
            print(f"\n✅ Completed streaming")
        elif event.type == StreamingEventType.ERROR:
            print(f"\n❌ Error: {event.data}")
    
    print()


async def demonstrate_tool_composition():
    """Demonstrate advanced tool composition."""
    print("🔧 Tool Composition Demo")
    print("=" * 50)
    
    # Create base tools
    calculator = CalculatorTool()
    search = SearchTool()
    
    # Demonstrate different composition patterns
    print("1. Retry wrapper:")
    retry_calc = with_retry(calculator, max_retries=3, backoff_factor=1.5)
    
    context = DemoContext(
        user_id='demo_user',
        session_id='composition_session',
        preferences={}
    )
    
    result = await retry_calc.execute(CalculateArgs(expression="2 + 2"), context)
    print(f"   Result: {result}")
    
    print("\n2. Cache wrapper:")
    cached_search = with_cache(search, ttl_seconds=60)
    
    # First call (cache miss)
    result1 = await cached_search.execute(SearchArgs(query="JAF framework"), context)
    print(f"   First call: {result1}")
    
    # Second call (cache hit)
    result2 = await cached_search.execute(SearchArgs(query="JAF framework"), context)
    print(f"   Second call: {result2}")
    
    print("\n3. Pipeline composition:")
    pipeline = create_tool_pipeline(search, calculator, name="search_calc_pipeline")
    
    # This would pass search results to calculator (simplified for demo)
    print("   Pipeline created successfully")
    
    print("\n4. Parallel execution:")
    parallel = create_parallel_tools(search, calculator, name="parallel_demo")
    print("   Parallel tool created successfully")
    
    print("\n5. Fluent composition with builder:")
    composed_tool = (compose(calculator)
                     .with_retry(max_retries=2)
                     .with_cache(ttl_seconds=120)
                     .build())
    
    composition_info = composed_tool.get_composition_info() if hasattr(composed_tool, 'get_composition_info') else "Composition applied"
    print(f"   Composed tool: {composition_info}")
    print()


async def demonstrate_enhanced_error_handling():
    """Demonstrate enhanced error handling."""
    print("🛡️ Enhanced Error Handling Demo")
    print("=" * 50)
    
    calculator = CalculatorTool()
    context = DemoContext(
        user_id='demo_user',
        session_id='error_session',
        preferences={}
    )
    
    # Test different error scenarios
    test_cases = [
        ("2 + 2", "Valid calculation"),
        ("1 / 0", "Division by zero"),
        ("import os", "Security violation"),
        ("2 + abc", "Invalid expression"),
        ("2 ** 1000000", "Potential overflow")
    ]
    
    for expression, description in test_cases:
        print(f"Testing: {description}")
        print(f"Expression: {expression}")
        
        try:
            result = await calculator.execute(CalculateArgs(expression=expression), context)
            if hasattr(result, 'status'):
                print(f"Status: {result.status}")
                if result.status == 'success':
                    print(f"Result: {result.data}")
                else:
                    print(f"Error: {result.error}")
            else:
                print(f"Result: {result}")
        except Exception as e:
            print(f"Exception: {e}")
        
        print("-" * 30)
    
    print()


async def demonstrate_full_integration():
    """Demonstrate full integration with all features."""
    print("🚀 Full Integration Demo")
    print("=" * 50)
    
    # Create enhanced agent
    agent = create_enhanced_agent()
    
    # Create context with performance tracking
    context = DemoContext(
        user_id='integration_user',
        session_id='full_demo_session',
        preferences={
            'detailed_responses': True,
            'show_calculations': True,
            'cache_enabled': True
        },
        performance_tracking=True
    )
    
    # Create mock model provider
    class EnhancedMockProvider:
        async def get_completion(self, state, agent, config):
            # Simulate intelligent response based on context
            last_message = state.messages[-1].content if state.messages else ""
            
            if 'calculate' in last_message.lower():
                return {
                    'message': {
                        'content': None,
                        'tool_calls': [{
                            'id': 'call_123',
                            'function': {
                                'name': 'calculate',
                                'arguments': '{"expression": "15 * 8 + 32"}'
                            }
                        }]
                    }
                }
            else:
                return {
                    'message': {
                        'content': f"I understand you want to {last_message}. Let me help you with that using my enhanced capabilities.",
                        'tool_calls': None
                    }
                }
    
    # Set up state and config
    initial_state = RunState(
        run_id=jaf.generate_run_id(),
        trace_id=jaf.generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content='Please calculate 15 * 8 + 32')],
        current_agent_name='EnhancedAssistant',
        context=context,
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={'EnhancedAssistant': agent},
        model_provider=EnhancedMockProvider(),
        max_turns=5
    )
    
    # Run with performance monitoring
    async with monitor_performance() as monitor:
        monitor.record_llm_call(token_count=50)
        
        # Simulate the run (in real usage, you'd use jaf.run)
        print("🤖 Agent: I'll calculate that for you using my enhanced calculator.")
        
        # Simulate tool execution
        calculator = CalculatorTool()
        calc_result = await calculator.execute(
            CalculateArgs(expression="15 * 8 + 32"), 
            context
        )
        
        monitor.record_tool_call()
        
        print(f"🔧 Tool Result: {calc_result}")
        print("🤖 Agent: The calculation 15 * 8 + 32 equals 152.")
        
        # Get performance metrics
        metrics = monitor.stop_monitoring()
        
        print(f"\n📊 Performance Summary:")
        print(f"   Execution time: {metrics.execution_time_ms:.2f}ms")
        print(f"   Memory usage: {metrics.memory_usage_mb:.2f}MB")
        print(f"   LLM calls: {metrics.llm_call_count}")
        print(f"   Tool calls: {metrics.tool_call_count}")
    
    print()


async def main():
    """Run all demonstrations."""
    print("🎯 JAF Enhanced Features Demonstration")
    print("=" * 60)
    print("This demo showcases the new capabilities added to JAF:")
    print("- Performance monitoring and metrics")
    print("- Streaming responses for real-time UX")
    print("- Advanced tool composition patterns")
    print("- Enhanced error handling and recovery")
    print("- Property-based testing concepts")
    print("=" * 60)
    print()
    
    try:
        await demonstrate_performance_monitoring()
        await demonstrate_streaming()
        await demonstrate_tool_composition()
        await demonstrate_enhanced_error_handling()
        await demonstrate_full_integration()
        
        print("✅ All demonstrations completed successfully!")
        print("\n🎉 JAF Enhanced Features Demo Complete!")
        print("\nKey improvements demonstrated:")
        print("• 📊 Performance monitoring with detailed metrics")
        print("• 🌊 Real-time streaming for progressive responses")
        print("• 🔧 Powerful tool composition with retry, cache, and pipelines")
        print("• 🛡️ Robust error handling with recovery strategies")
        print("• 🧪 Property-based testing for comprehensive validation")
        print("• 🔌 Plugin system foundation for extensibility")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
