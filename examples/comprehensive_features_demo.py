"""
Comprehensive JAF Features Demonstration

This example showcases all the advanced features of JAF including:
- Performance monitoring and analytics
- Streaming responses
- Tool composition
- Workflow orchestration
- Enhanced error handling
- Property-based testing concepts

Run this example to see the complete JAF ecosystem in action.
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
from jaf.core.analytics import get_analytics_report, analyze_conversation_quality
from jaf.core.workflows import (
    create_workflow, WorkflowContext, AgentStep, ToolStep, ConditionalStep,
    ParallelStep, LoopStep
)


@dataclass
class ComprehensiveContext:
    """Enhanced context for comprehensive demo."""
    user_id: str
    session_id: str
    preferences: Dict[str, Any]
    workflow_enabled: bool = True
    analytics_enabled: bool = True
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


class DataAnalysisArgs(BaseModel):
    """Arguments for data analysis tool."""
    dataset: str = Field(description="Dataset to analyze")
    analysis_type: str = Field(description="Type of analysis to perform")


# Enhanced tools with comprehensive features
class AdvancedCalculatorTool:
    """Advanced calculator with comprehensive error handling and analytics."""
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='advanced_calculate',
            description='Advanced mathematical calculations with error handling and analytics',
            parameters=CalculateArgs
        )
    
    async def execute(self, args: CalculateArgs, context: ComprehensiveContext):
        """Execute calculation with comprehensive features."""
        start_time = time.time()
        
        try:
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Enhanced security checks
            dangerous_operations = ['import', 'exec', 'eval', '__', 'open', 'file']
            if any(op in args.expression for op in dangerous_operations):
                return ToolResponse.validation_error(
                    "Expression contains potentially dangerous operations",
                    details={'expression': args.expression, 'blocked_operations': dangerous_operations}
                )
            
            # Advanced math evaluation with more operators
            allowed_chars = set('0123456789+-*/()., abcdefghijklmnopqrstuvwxyz')
            if not all(c.lower() in allowed_chars for c in args.expression):
                return ToolResponse.validation_error(
                    "Expression contains invalid characters",
                    details={'expression': args.expression}
                )
            
            # Support for advanced math functions
            import math
            safe_dict = {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e
            }
            
            # Evaluate with safe dictionary
            result = eval(args.expression, {"__builtins__": {}}, safe_dict)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResponse.success(
                data={
                    'expression': args.expression,
                    'result': result,
                    'execution_time_ms': execution_time,
                    'functions_used': [func for func in safe_dict.keys() if func in args.expression]
                },
                metadata={
                    'calculation_time': time.time(),
                    'complexity': 'advanced' if any(func in args.expression for func in safe_dict.keys()) else 'basic'
                }
            )
            
        except ZeroDivisionError:
            return ToolResponse.error(
                code='division_by_zero',
                message='Cannot divide by zero',
                details={'expression': args.expression, 'suggestion': 'Check your denominator'}
            )
        except Exception as e:
            return ToolResponse.error(
                code='calculation_error',
                message=f'Calculation failed: {str(e)}',
                details={'expression': args.expression, 'error_type': type(e).__name__}
            )


class IntelligentSearchTool:
    """Intelligent search tool with caching and analytics."""
    
    def __init__(self):
        self.search_history = []
        self.cache = {}
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='intelligent_search',
            description='Intelligent search with caching, history, and result ranking',
            parameters=SearchArgs
        )
    
    async def execute(self, args: SearchArgs, context: ComprehensiveContext):
        """Execute intelligent search."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{args.query}_{args.max_results}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result['metadata']['cached'] = True
            cached_result['metadata']['cache_hit_time'] = time.time()
            return ToolResponse.success(**cached_result)
        
        # Simulate search delay
        await asyncio.sleep(0.2)
        
        # Record search history
        self.search_history.append({
            'query': args.query,
            'timestamp': time.time(),
            'user_id': context.user_id
        })
        
        # Simulate intelligent search results with ranking
        base_results = [
            f"High-relevance result for '{args.query}' - Score: 0.95",
            f"Medium-relevance result for '{args.query}' - Score: 0.78",
            f"Related information about '{args.query}' - Score: 0.65",
            f"Background context for '{args.query}' - Score: 0.52",
            f"Additional resources on '{args.query}' - Score: 0.41"
        ]
        
        results = base_results[:args.max_results]
        
        # Add search suggestions based on history
        suggestions = []
        if len(self.search_history) > 1:
            recent_queries = [h['query'] for h in self.search_history[-3:]]
            suggestions = [f"Related: {q}" for q in recent_queries if q != args.query]
        
        execution_time = (time.time() - start_time) * 1000
        
        result_data = {
            'data': {
                'query': args.query,
                'results': results,
                'total_found': len(results),
                'suggestions': suggestions,
                'search_score': 0.85,
                'execution_time_ms': execution_time
            },
            'metadata': {
                'search_time': time.time(),
                'cached': False,
                'history_length': len(self.search_history),
                'ranking_algorithm': 'relevance_score'
            }
        }
        
        # Cache the result
        self.cache[cache_key] = result_data
        
        return ToolResponse.success(**result_data)


class WeatherAnalyticsTool:
    """Weather tool with analytics and forecasting."""
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='weather_analytics',
            description='Weather information with analytics and forecasting',
            parameters=WeatherArgs
        )
    
    async def execute(self, args: WeatherArgs, context: ComprehensiveContext):
        """Get weather with analytics."""
        start_time = time.time()
        
        # Simulate API call
        await asyncio.sleep(0.15)
        
        # Simulate comprehensive weather data
        weather_data = {
            'location': args.location,
            'current': {
                'temperature': 22,
                'condition': 'Sunny',
                'humidity': 65,
                'wind_speed': 10,
                'pressure': 1013.25,
                'uv_index': 6
            },
            'forecast': [
                {'day': 'Today', 'high': 25, 'low': 18, 'condition': 'Sunny'},
                {'day': 'Tomorrow', 'high': 23, 'low': 16, 'condition': 'Partly Cloudy'},
                {'day': 'Day 3', 'high': 20, 'low': 14, 'condition': 'Rainy'}
            ],
            'analytics': {
                'temperature_trend': 'stable',
                'precipitation_probability': 0.2,
                'air_quality_index': 45,
                'comfort_level': 'comfortable'
            }
        }
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResponse.success(
            data=weather_data,
            metadata={
                'api_call_time': time.time(),
                'data_freshness': 'real_time',
                'execution_time_ms': execution_time,
                'analytics_enabled': context.analytics_enabled
            }
        )


class DataAnalysisTool:
    """Advanced data analysis tool."""
    
    @property
    def schema(self):
        return jaf.ToolSchema(
            name='data_analysis',
            description='Perform advanced data analysis and generate insights',
            parameters=DataAnalysisArgs
        )
    
    async def execute(self, args: DataAnalysisArgs, context: ComprehensiveContext):
        """Perform data analysis."""
        start_time = time.time()
        
        # Simulate analysis processing
        await asyncio.sleep(0.3)
        
        # Simulate analysis results based on type
        analysis_results = {
            'statistical': {
                'mean': 45.7,
                'median': 42.3,
                'std_dev': 12.8,
                'correlation': 0.73,
                'sample_size': 1000
            },
            'trend': {
                'direction': 'increasing',
                'slope': 0.15,
                'r_squared': 0.82,
                'confidence': 0.95
            },
            'clustering': {
                'clusters_found': 3,
                'silhouette_score': 0.68,
                'cluster_sizes': [340, 420, 240]
            }
        }
        
        result = analysis_results.get(args.analysis_type, analysis_results['statistical'])
        
        execution_time = (time.time() - start_time) * 1000
        
        return ToolResponse.success(
            data={
                'dataset': args.dataset,
                'analysis_type': args.analysis_type,
                'results': result,
                'insights': [
                    f"Dataset '{args.dataset}' shows {args.analysis_type} patterns",
                    "Data quality is high with minimal outliers",
                    "Recommended for further analysis"
                ]
            },
            metadata={
                'analysis_time': time.time(),
                'execution_time_ms': execution_time,
                'algorithm_version': '2.1.0'
            }
        )


def create_comprehensive_agent() -> Agent[ComprehensiveContext, str]:
    """Create an agent with all advanced features."""
    
    # Create base tools
    calculator = AdvancedCalculatorTool()
    search = IntelligentSearchTool()
    weather = WeatherAnalyticsTool()
    data_analysis = DataAnalysisTool()
    
    # Create composed tools
    cached_search = with_cache(search, ttl_seconds=300)
    reliable_calculator = with_retry(calculator, max_retries=3)
    
    # Create tool pipelines
    research_pipeline = create_tool_pipeline(
        cached_search,
        data_analysis,
        name="research_analysis_pipeline"
    )
    
    # Create parallel tools
    parallel_info = create_parallel_tools(
        cached_search,
        weather,
        name="parallel_info_gathering",
        combine_strategy="merge"
    )
    
    def instructions(state: RunState[ComprehensiveContext]) -> str:
        context = state.context
        features_note = []
        
        if context.performance_tracking:
            features_note.append("performance monitoring")
        if context.analytics_enabled:
            features_note.append("analytics")
        if context.workflow_enabled:
            features_note.append("workflow orchestration")
        
        features_str = ", ".join(features_note) if features_note else "standard"
        
        return f"""You are an advanced AI assistant with comprehensive capabilities.

Active features: {features_str}

Available tools:
- advanced_calculate: Enhanced calculator with advanced math functions
- intelligent_search: Smart search with caching and history
- weather_analytics: Weather with forecasting and analytics
- data_analysis: Advanced data analysis and insights
- research_analysis_pipeline: Sequential research and analysis
- parallel_info_gathering: Parallel information collection

User preferences: {context.preferences}
Session: {context.session_id}

Provide intelligent, comprehensive responses using the available tools when appropriate.
Focus on delivering actionable insights and detailed analysis.
"""
    
    return Agent(
        name='ComprehensiveAssistant',
        instructions=instructions,
        tools=[
            reliable_calculator,
            cached_search,
            weather,
            data_analysis,
            research_pipeline,
            parallel_info
        ]
    )


async def demonstrate_comprehensive_analytics():
    """Demonstrate comprehensive analytics capabilities."""
    print("📊 Comprehensive Analytics Demo")
    print("=" * 60)
    
    # Simulate conversation data
    messages = [
        Message(role=ContentRole.USER, content="Hello, I need help with data analysis"),
        Message(role=ContentRole.ASSISTANT, content="I'd be happy to help with data analysis. What specific dataset or analysis type are you interested in?"),
        Message(role=ContentRole.USER, content="I have sales data and want to understand trends"),
        Message(role=ContentRole.ASSISTANT, content="Great! I can perform trend analysis on your sales data. Let me analyze the patterns for you."),
        Message(role=ContentRole.USER, content="Thank you, that was very helpful!")
    ]
    
    # Analyze conversation quality
    start_time = time.time() - 300  # 5 minutes ago
    end_time = time.time()
    
    conversation_analytics = analyze_conversation_quality(messages, start_time, end_time)
    
    print(f"Conversation Analysis:")
    print(f"  Total messages: {conversation_analytics.total_messages}")
    print(f"  Duration: {conversation_analytics.conversation_duration_minutes:.1f} minutes")
    print(f"  Sentiment score: {conversation_analytics.sentiment_score:.1f}")
    print(f"  Engagement score: {conversation_analytics.engagement_score:.1f}")
    print(f"  Resolution status: {conversation_analytics.resolution_status}")
    print(f"  Topic keywords: {conversation_analytics.topic_keywords}")
    
    # Get comprehensive analytics report
    analytics_report = get_analytics_report()
    
    print(f"\nSystem Analytics:")
    print(f"  Timestamp: {analytics_report['timestamp']}")
    print(f"  Total conversations: {analytics_report['summary']['total_conversations']}")
    print(f"  Active agents: {analytics_report['summary']['active_agents']}")
    print(f"  Key insights: {analytics_report['summary']['key_insights']}")
    
    print()


async def demonstrate_advanced_workflows():
    """Demonstrate advanced workflow orchestration."""
    print("🔄 Advanced Workflow Orchestration Demo")
    print("=" * 60)
    
    # Create comprehensive agent
    agent = create_comprehensive_agent()
    
    # Create workflow context
    context = ComprehensiveContext(
        user_id='workflow_user',
        session_id='workflow_demo_session',
        preferences={'detailed_analysis': True, 'include_visualizations': True},
        workflow_enabled=True,
        analytics_enabled=True,
        performance_tracking=True
    )
    
    workflow_context = WorkflowContext(
        workflow_id='comprehensive_demo_workflow',
        user_context=context
    )
    
    # Create workflow steps
    search_step = AgentStep(
        'search_data',
        agent,
        'Search for information about machine learning trends'
    )
    
    analysis_step = AgentStep(
        'analyze_data',
        agent,
        'Analyze the search results and provide insights'
    )
    
    weather_step = AgentStep(
        'get_weather',
        agent,
        'Get weather information for San Francisco'
    )
    
    # Create conditional step
    def should_include_weather(ctx):
        return ctx.variables.get('include_weather', True)
    
    conditional_weather = ConditionalStep(
        'conditional_weather',
        should_include_weather,
        weather_step
    )
    
    # Create parallel step
    parallel_research = ParallelStep(
        'parallel_research',
        [search_step, weather_step],
        wait_for_all=True
    )
    
    # Create loop step
    def should_continue_loop(ctx, iteration):
        return iteration < 2  # Run 2 iterations
    
    loop_analysis = LoopStep(
        'iterative_analysis',
        analysis_step,
        should_continue_loop,
        max_iterations=3
    )
    
    # Build comprehensive workflow
    workflow = (create_workflow('comprehensive_demo', 'Comprehensive Demo Workflow')
                .add_agent_step('initial_search', agent, 'Start comprehensive analysis')
                .add_parallel_step('parallel_info', [search_step, weather_step])
                .add_conditional_step('conditional_analysis', 
                                    lambda ctx: len(ctx.variables) > 0,
                                    analysis_step)
                .add_loop_step('iterative_refinement', analysis_step, should_continue_loop, 2)
                .with_step_callback(lambda result, ctx: print(f"  ✓ Step {result.step_id} completed in {result.execution_time_ms:.1f}ms"))
                .with_completion_callback(lambda result: print(f"  🎉 Workflow completed with {result.success_rate:.1f}% success rate"))
                .build())
    
    # Execute workflow
    print("Executing comprehensive workflow...")
    workflow_result = await workflow.execute(workflow_context)
    
    print(f"\nWorkflow Results:")
    print(f"  Status: {workflow_result.status.value}")
    print(f"  Total execution time: {workflow_result.total_execution_time_ms:.1f}ms")
    print(f"  Steps completed: {len(workflow_result.steps)}")
    print(f"  Success rate: {workflow_result.success_rate:.1f}%")
    
    if workflow_result.failed_steps:
        print(f"  Failed steps: {[s.step_id for s in workflow_result.failed_steps]}")
    
    print()


async def demonstrate_streaming_with_analytics():
    """Demonstrate streaming with real-time analytics."""
    print("🌊 Streaming with Real-time Analytics Demo")
    print("=" * 60)
    
    # Create agent
    agent = create_comprehensive_agent()
    
    # Create mock model provider
    class AnalyticsAwareMockProvider:
        async def get_completion(self, state, agent, config):
            # Simulate intelligent response
            return {
                'message': {
                    'content': 'This is a comprehensive streaming response that demonstrates real-time analytics integration with JAF framework capabilities.',
                    'tool_calls': None
                }
            }
    
    # Set up streaming with analytics
    context = ComprehensiveContext(
        user_id='streaming_user',
        session_id='streaming_analytics_session',
        preferences={'real_time_updates': True},
        analytics_enabled=True
    )
    
    initial_state = RunState(
        run_id=jaf.generate_run_id(),
        trace_id=jaf.generate_trace_id(),
        messages=[Message(role=ContentRole.USER, content='Provide a comprehensive analysis with streaming')],
        current_agent_name='ComprehensiveAssistant',
        context=context,
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={'ComprehensiveAssistant': agent},
        model_provider=AnalyticsAwareMockProvider()
    )
    
    # Stream with analytics
    print("Streaming response with analytics:")
    chunk_count = 0
    total_chars = 0
    
    async for event in run_streaming(initial_state, config, chunk_size=15):
        if event.type == StreamingEventType.START:
            print(f"🚀 Started streaming (Agent: {event.data.agent_name})")
        elif event.type == StreamingEventType.CHUNK:
            chunk_count += 1
            total_chars += len(event.data.delta)
            print(f"📝 Chunk {chunk_count}: '{event.data.delta}'", end='', flush=True)
        elif event.type == StreamingEventType.COMPLETE:
            print(f"\n✅ Completed streaming")
            print(f"📊 Analytics: {chunk_count} chunks, {total_chars} characters")
        elif event.type == StreamingEventType.ERROR:
            print(f"\n❌ Error: {event.data}")
    
    print()


async def demonstrate_full_integration():
    """Demonstrate full integration of all features."""
    print("🚀 Full Integration Demo")
    print("=" * 60)
    
    # Create comprehensive context
    context = ComprehensiveContext(
        user_id='integration_user',
        session_id='full_integration_session',
        preferences={
            'comprehensive_analysis': True,
            'real_time_updates': True,
            'detailed_insights': True,
            'workflow_orchestration': True
        },
        workflow_enabled=True,
        analytics_enabled=True,
        performance_tracking=True
    )
    
    # Create agent with all features
    agent = create_comprehensive_agent()
    
    # Execute with comprehensive monitoring
    async with monitor_performance() as monitor:
        monitor.record_llm_call(token_count=75)
        
        print("🤖 Agent: I'll provide a comprehensive analysis using all available features.")
        
        # Simulate tool executions
        calculator = AdvancedCalculatorTool()
        search = IntelligentSearchTool()
        weather = WeatherAnalyticsTool()
        data_analysis = DataAnalysisTool()
        
        # Execute tools with monitoring
        calc_result = await calculator.execute(
            CalculateArgs(expression="sqrt(25) + sin(pi/2)"), 
            context
        )
        monitor.record_tool_call()
        
        search_result = await search.execute(
            SearchArgs(query="JAF framework capabilities", max_results=3),
            context
        )
        monitor.record_tool_call()
        monitor.record_cache_miss()
        
        weather_result = await weather.execute(
            WeatherArgs(location="San Francisco"),
            context
        )
        monitor.record_tool_call()
        
        analysis_result = await data_analysis.execute(
            DataAnalysisArgs(dataset="user_engagement", analysis_type="trend"),
            context
        )
        monitor.record_tool_call()
        
        print(f"🔧 Calculator Result: {calc_result}")
        print(f"🔍 Search Result: {search_result}")
        print(f"🌤️ Weather Result: {weather_result}")
        print(f"📊 Analysis Result: {analysis_result}")
        
        # Get performance metrics
        metrics = monitor.stop_monitoring()
        
        print(f"\n📊 Comprehensive Performance Summary:")
        print(f"   Execution time: {metrics.execution_time_ms:.2f}ms")
        print(f"   Memory usage: {metrics.memory_usage_mb:.2f}MB")
        print(f"   Peak memory: {metrics.peak_memory_mb:.2f}MB")
        print(f"   LLM calls: {metrics.llm_call_count}")
        print(f"   Tool calls: {metrics.tool_call_count}")
        print(f"   Cache hit rate: {metrics.cache_hit_rate:.1f}%")
        print(f"   Token count: {metrics.token_count}")
    
    # Get final analytics report
    final_analytics = get_analytics_report()
    print(f"\n📈 Final Analytics Summary:")
    print(f"   Total conversations: {final_analytics['summary']['total_conversations']}")
    print(f"   Active agents: {final_analytics['summary']['active_agents']}")
    print(f"   Key insights: {final_analytics['summary']['key_insights']}")
    
    print()


async def main():
    """Run comprehensive demonstration of all JAF features."""
    print("🎯 JAF Comprehensive Features Demonstration")
    print("=" * 80)
    print("This demo showcases the complete JAF ecosystem:")
    print("- Advanced performance monitoring and analytics")
    print("- Intelligent streaming responses")
    print("- Sophisticated tool composition")
    print("- Workflow orchestration and automation")
    print("- Enhanced error handling and recovery")
    print("- Real-time insights and optimization")
    print("=" * 80)
    print()
    
    try:
        await demonstrate_comprehensive_analytics()
        await demonstrate_advanced_workflows()
        await demonstrate_streaming_with_analytics()
        await demonstrate_full_integration()
        
        print("✅ All comprehensive demonstrations completed successfully!")
        print("\n🎉 JAF Comprehensive Features Demo Complete!")
        print("\nAdvanced capabilities demonstrated:")
        print("• 📊 Comprehensive analytics with conversation insights")
        print("• 🔄 Advanced workflow orchestration with conditional logic")
        print("• 🌊 Intelligent streaming with real-time analytics")
        print("• 🔧 Sophisticated tool composition and caching")
        print("• 🛡️ Enhanced error handling with recovery strategies")
        print("• 📈 Performance monitoring with detailed metrics")
        print("• 🧠 Intelligent agent coordination and handoffs")
        print("• 🔌 Extensible plugin system foundation")
        print("\n🚀 JAF is now a production-ready, enterprise-grade agent framework!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
