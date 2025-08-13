"""
Iterative Search Agent Example - Showcasing the ADK Callback System

This example demonstrates the power of the comprehensive callback system by 
implementing a sophisticated "ReAct" style agent that iteratively searches,
accumulates context, checks for synthesis completion, and refines its queries
until it has enough information to provide a comprehensive answer.

Key Features Demonstrated:
- Iterative information gathering with synthesis checking
- Dynamic query rewriting based on accumulated context
- Context accumulation and management
- Loop detection to prevent repetitive searches
- Custom LLM call modification for focused searches
- Synthesis-based termination with confidence scoring

This showcases how the callback system transforms the ADK runner from a simple
executor into a sophisticated reasoning engine capable of complex agent patterns.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import JAF core components
from jaf.core.types import Agent, Message, Tool, ToolSchema
from jaf.core.tool_results import ToolResponse, ToolResult
from jaf.providers.model import make_litellm_provider

# Import ADK with callback system
from adk.runners import (
    run_agent, 
    RunnerConfig, 
    RunnerCallbacks,
    LLMControlResult,
    SynthesisCheckResult,
    IterationControlResult
)
from pydantic import BaseModel


# ========== Search Tool Implementation ==========

class SearchArgs(BaseModel):
    query: str
    max_results: int = 5


class WebSearchTool:
    """
    Mock web search tool that simulates searching for information.
    
    In a real implementation, this would integrate with actual search APIs
    like Google Search, Bing, or specialized databases.
    """
    
    @property
    def schema(self):
        return ToolSchema(
            name="web_search",
            description="Search the web for information on a given topic",
            parameters=SearchArgs
        )
    
    async def execute(self, args: SearchArgs, context: Any) -> ToolResult:
        """
        Simulate web search by returning mock results based on the query.
        
        The mock results are designed to require multiple searches to get
        complete information, demonstrating the iterative nature.
        """
        query_lower = args.query.lower()
        
        # Mock search results based on query content
        if "machine learning" in query_lower:
            if "healthcare" in query_lower or "medical" in query_lower:
                results = [
                    {
                        "title": "AI in Medical Diagnosis",
                        "content": "Machine learning algorithms are revolutionizing medical diagnosis, with deep learning models achieving 95% accuracy in radiology.",
                        "url": "https://example.com/ml-medical-diagnosis",
                        "snippet": "Deep learning applications in medical imaging and diagnosis"
                    },
                    {
                        "title": "Drug Discovery with ML",
                        "content": "Pharmaceutical companies use ML to accelerate drug discovery, reducing development time from 10-15 years to 5-7 years.",
                        "url": "https://example.com/ml-drug-discovery",
                        "snippet": "Machine learning accelerates pharmaceutical research"
                    }
                ]
            elif "finance" in query_lower or "trading" in query_lower:
                results = [
                    {
                        "title": "Algorithmic Trading Systems",
                        "content": "ML-powered trading algorithms process millions of data points per second to make investment decisions.",
                        "url": "https://example.com/ml-trading",
                        "snippet": "High-frequency trading with machine learning"
                    },
                    {
                        "title": "Credit Risk Assessment",
                        "content": "Banks use ML models for credit scoring, reducing default rates by 30% compared to traditional methods.",
                        "url": "https://example.com/ml-credit-risk",
                        "snippet": "ML improves credit risk evaluation accuracy"
                    }
                ]
            else:
                results = [
                    {
                        "title": "Introduction to Machine Learning",
                        "content": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
                        "url": "https://example.com/ml-intro",
                        "snippet": "Basic overview of machine learning concepts"
                    },
                    {
                        "title": "Types of Machine Learning",
                        "content": "ML includes supervised, unsupervised, and reinforcement learning approaches.",
                        "url": "https://example.com/ml-types",
                        "snippet": "Categorization of ML approaches"
                    }
                ]
        elif "artificial intelligence" in query_lower or "ai applications" in query_lower:
            results = [
                {
                    "title": "AI Applications Across Industries",
                    "content": "AI is transforming industries including healthcare, finance, transportation, and entertainment.",
                    "url": "https://example.com/ai-applications",
                    "snippet": "Cross-industry AI implementation overview"
                },
                {
                    "title": "Future of AI Technology",
                    "content": "Emerging AI trends include AGI development, quantum-AI hybrid systems, and ethical AI frameworks.",
                    "url": "https://example.com/ai-future",
                    "snippet": "Upcoming developments in artificial intelligence"
                }
            ]
        else:
            # Generic results for other queries
            results = [
                {
                    "title": f"Search Results for: {args.query}",
                    "content": f"General information about {args.query} and related topics.",
                    "url": f"https://example.com/search?q={args.query.replace(' ', '+')}",
                    "snippet": f"Overview of {args.query}"
                }
            ]
        
        # Limit results to requested maximum
        results = results[:args.max_results]
        
        # Return in the format expected by the callback system
        return ToolResponse.success(
            {
                "results": results,
                "query": args.query,
                "total_found": len(results),
                # Include contexts for accumulation
                "contexts": [
                    {
                        "id": f"search_{hash(args.query)}_{i}",
                        "source": "web_search",
                        "query": args.query,
                        "content": result["content"],
                        "title": result["title"],
                        "url": result["url"],
                        "relevance": 0.9 - (i * 0.1)  # Decreasing relevance
                    }
                    for i, result in enumerate(results)
                ]
            }
        )


# ========== Iterative Search Callbacks Implementation ==========

class IterativeSearchCallbacks:
    """
    Sophisticated callback implementation for iterative search and synthesis.
    
    This class demonstrates how to use the callback system to implement
    complex agent behaviors like ReAct (Reasoning + Acting) patterns.
    """
    
    def __init__(self, max_iterations: int = 5, synthesis_threshold: int = 3):
        self.max_iterations = max_iterations
        self.synthesis_threshold = synthesis_threshold
        self.iteration_count = 0
        self.context_accumulator = []
        self.search_history = []
        self.original_query = ""
        self.synthesis_confidence = 0.0
        
        print("🔄 Initializing Iterative Search Agent with:")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Synthesis threshold: {synthesis_threshold} context items")
    
    async def on_start(self, context, message, session_state):
        """Initialize the iterative search session."""
        self.original_query = message.content
        print(f"🚀 Starting iterative search for: '{self.original_query}'")
        print("="*60)
    
    async def on_iteration_start(self, iteration: int) -> Optional[IterationControlResult]:
        """Control iteration flow and provide progress updates."""
        self.iteration_count = iteration
        print(f"\n🔄 ITERATION {iteration}/{self.max_iterations}")
        print("-" * 40)
        
        if iteration > self.max_iterations:
            print(f"⏹️  Maximum iterations ({self.max_iterations}) reached. Stopping.")
            return {'continue_iteration': False}
        
        return None
    
    async def on_query_rewrite(self, original_query: str, context_data: List[Any]) -> Optional[str]:
        """
        Dynamically rewrite queries based on accumulated context.
        
        This is a key component of iterative search - using previous results
        to inform and refine subsequent searches.
        """
        if not context_data:
            return None
        
        print(f"🧠 Analyzing {len(context_data)} context items for query refinement...")
        
        # Analyze context to identify gaps or areas needing more specific information
        topics_covered = set()
        for item in context_data:
            content = item.get('content', '').lower()
            if 'healthcare' in content or 'medical' in content:
                topics_covered.add('healthcare')
            if 'finance' in content or 'trading' in content:
                topics_covered.add('finance')
            if 'diagnosis' in content:
                topics_covered.add('diagnosis')
            if 'drug' in content:
                topics_covered.add('pharmaceuticals')
        
        # Generate refined query based on gaps
        if 'machine learning' in original_query.lower():
            if not topics_covered:
                refined_query = f"{original_query} applications in different industries"
            elif 'healthcare' in topics_covered and 'finance' not in topics_covered:
                refined_query = f"{original_query} applications in finance and trading"
            elif 'finance' in topics_covered and 'healthcare' not in topics_covered:
                refined_query = f"{original_query} applications in healthcare and medicine"
            elif len(topics_covered) >= 2:
                refined_query = f"{original_query} future trends and emerging applications"
            else:
                refined_query = f"{original_query} real-world case studies and examples"
        else:
            refined_query = f"{original_query} detailed analysis and applications"
        
        if refined_query != original_query:
            print(f"📝 Query refined: '{original_query}' → '{refined_query}'")
            print(f"   Based on coverage: {', '.join(topics_covered) if topics_covered else 'none yet'}")
            return refined_query
        
        return None
    
    async def on_before_llm_call(self, agent, message, session_state) -> Optional[LLMControlResult]:
        """
        Modify LLM calls to provide context and improve search decisions.
        
        This demonstrates how to inject accumulated context into LLM calls
        to make more informed decisions about what to search for next.
        """
        if self.context_accumulator and self.iteration_count > 1:
            # Create enriched prompt with context summary
            context_summary = self._create_context_summary()
            
            enriched_content = f"""Based on previous search results:
{context_summary}

Original query: {self.original_query}
Current search iteration: {self.iteration_count}

{message.content}

Please search for information that fills gaps in our current knowledge."""
            
            enriched_message = Message(role='user', content=enriched_content)
            
            print(f"💡 Enriching LLM call with context from {len(self.context_accumulator)} previous results")
            
            return {'message': enriched_message}
        
        return None
    
    async def on_tool_selected(self, tool_name: Optional[str], params: Any) -> None:
        """Track search patterns and provide feedback."""
        if tool_name == "web_search":
            query = params.get('query', 'unknown') if params else 'unknown'
            print(f"🔍 Executing search: '{query}'")
            
            # Track search history for loop detection
            self.search_history.append({
                'iteration': self.iteration_count,
                'query': query,
                'timestamp': asyncio.get_event_loop().time()
            })
    
    async def on_loop_detection(self, tool_history: List[Dict[str, Any]], current_tool: str) -> bool:
        """
        Detect and prevent repetitive searches.
        
        This prevents the agent from getting stuck in loops by searching
        for the same or very similar information repeatedly.
        """
        if current_tool != "web_search" or len(tool_history) < 2:
            return False
        
        # Check for recent similar queries
        recent_queries = [item.get('params', {}).get('query', '') for item in tool_history[-3:]]
        current_query = ""
        
        # Simple similarity check (in production, would use more sophisticated methods)
        for query in recent_queries:
            if query and current_query:
                similarity = len(set(query.lower().split()) & set(current_query.lower().split())) / max(len(query.split()), len(current_query.split()))
                if similarity > 0.7:
                    print(f"🚫 Loop detected: Similar query '{query}' recently executed")
                    return True
        
        return False
    
    async def on_context_update(self, current_context: List[Any], new_items: List[Any]) -> Optional[List[Any]]:
        """
        Manage context accumulation with filtering and prioritization.
        
        This demonstrates sophisticated context management, including
        deduplication, relevance filtering, and size management.
        """
        print(f"📚 Adding {len(new_items)} new context items...")
        
        # Add new items to accumulator
        for item in new_items:
            # Simple deduplication based on content similarity
            is_duplicate = any(
                item.get('content', '')[:100] == existing.get('content', '')[:100]
                for existing in self.context_accumulator
            )
            
            if not is_duplicate:
                self.context_accumulator.append(item)
        
        # Sort by relevance and keep top items
        self.context_accumulator.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        self.context_accumulator = self.context_accumulator[:20]  # Keep top 20 items
        
        print(f"   Total context items: {len(self.context_accumulator)}")
        print(f"   Unique sources: {len(set(item.get('source', 'unknown') for item in self.context_accumulator))}")
        
        return self.context_accumulator
    
    async def on_check_synthesis(self, session_state: Dict[str, Any], context_data: List[Any]) -> Optional[SynthesisCheckResult]:
        """
        Determine if enough information has been gathered for synthesis.
        
        This is the heart of the iterative pattern - deciding when to stop
        searching and provide a final answer based on accumulated knowledge.
        """
        if len(context_data) < self.synthesis_threshold:
            print(f"📊 Synthesis check: {len(context_data)}/{self.synthesis_threshold} items - continuing search")
            return None
        
        print(f"🧮 Evaluating synthesis readiness with {len(context_data)} context items...")
        
        # Analyze context coverage and quality
        coverage_score = self._analyze_coverage(context_data)
        quality_score = self._analyze_quality(context_data)
        completeness_score = min(len(context_data) / 10.0, 1.0)  # Up to 10 items for full score
        
        # Calculate overall confidence
        self.synthesis_confidence = (coverage_score + quality_score + completeness_score) / 3.0
        
        print(f"   Coverage: {coverage_score:.2f}")
        print(f"   Quality: {quality_score:.2f}")
        print(f"   Completeness: {completeness_score:.2f}")
        print(f"   Overall confidence: {self.synthesis_confidence:.2f}")
        
        # Complete synthesis if confidence is high enough
        if self.synthesis_confidence >= 0.75:
            synthesis_answer = self._generate_synthesis_prompt(context_data)
            print(f"✅ Synthesis complete! Confidence: {self.synthesis_confidence:.2f}")
            
            return {
                'complete': True,
                'answer': synthesis_answer,
                'confidence': self.synthesis_confidence
            }
        else:
            print(f"🔄 Synthesis confidence too low ({self.synthesis_confidence:.2f} < 0.75) - continuing search")
            return None
    
    async def on_complete(self, response) -> None:
        """Provide completion summary and statistics."""
        print("\n" + "="*60)
        print("🎉 ITERATIVE SEARCH COMPLETED")
        print(f"   Total iterations: {self.iteration_count}")
        print(f"   Context items gathered: {len(self.context_accumulator)}")
        print(f"   Searches performed: {len(self.search_history)}")
        print(f"   Final confidence: {self.synthesis_confidence:.2f}")
        print(f"   Execution time: {response.execution_time_ms:.0f}ms")
        print("="*60)
    
    async def on_error(self, error: Exception, context) -> None:
        """Handle errors gracefully with context preservation."""
        print(f"❌ Error in iteration {self.iteration_count}: {str(error)}")
        print(f"   Context preserved: {len(self.context_accumulator)} items")
    
    # ========== Helper Methods ==========
    
    def _create_context_summary(self) -> str:
        """Create a concise summary of accumulated context."""
        if not self.context_accumulator:
            return "No context available yet."
        
        summaries = []
        for item in self.context_accumulator[:5]:  # Top 5 items
            title = item.get('title', 'Unknown')
            content = item.get('content', '')[:100] + "..." if len(item.get('content', '')) > 100 else item.get('content', '')
            summaries.append(f"- {title}: {content}")
        
        return "\n".join(summaries)
    
    def _analyze_coverage(self, context_data: List[Any]) -> float:
        """Analyze how well the context covers different aspects of the query."""
        topics = set()
        for item in context_data:
            content = item.get('content', '').lower()
            # Extract key topics (simplified)
            if any(word in content for word in ['healthcare', 'medical', 'health']):
                topics.add('healthcare')
            if any(word in content for word in ['finance', 'trading', 'bank', 'investment']):
                topics.add('finance')
            if any(word in content for word in ['technology', 'software', 'system']):
                topics.add('technology')
            if any(word in content for word in ['research', 'study', 'analysis']):
                topics.add('research')
        
        # More topics covered = better coverage
        return min(len(topics) / 4.0, 1.0)
    
    def _analyze_quality(self, context_data: List[Any]) -> float:
        """Analyze the quality of gathered context."""
        if not context_data:
            return 0.0
        
        total_relevance = sum(item.get('relevance', 0.5) for item in context_data)
        avg_relevance = total_relevance / len(context_data)
        
        # Factor in content length (longer content often means more detailed)
        avg_content_length = sum(len(item.get('content', '')) for item in context_data) / len(context_data)
        length_score = min(avg_content_length / 200.0, 1.0)  # Normalize to 200 chars
        
        return (avg_relevance + length_score) / 2.0
    
    def _generate_synthesis_prompt(self, context_data: List[Any]) -> str:
        """Generate a comprehensive prompt for final synthesis."""
        context_summary = []
        for i, item in enumerate(context_data[:10], 1):
            title = item.get('title', f'Source {i}')
            content = item.get('content', '')
            context_summary.append(f"{i}. {title}\n   {content}")
        
        return f"""Based on the comprehensive research conducted across {self.iteration_count} iterations, 
please provide a detailed and well-structured answer to: "{self.original_query}"

Use the following gathered information:

{chr(10).join(context_summary)}

Please synthesize this information into a comprehensive response that addresses all aspects of the original query."""


# ========== Example Usage ==========

async def create_iterative_search_agent() -> Agent:
    """Create an agent configured for iterative search."""
    def instructions(state):
        return """You are an advanced research agent capable of iterative information gathering.

Your capabilities:
- Conduct thorough web searches on any topic
- Analyze and synthesize information from multiple sources
- Iteratively refine search queries based on findings
- Provide comprehensive, well-researched answers

When given a query:
1. Search for initial information
2. Analyze what you've found and identify gaps
3. Conduct additional targeted searches to fill gaps
4. Continue until you have comprehensive coverage
5. Synthesize all findings into a complete answer

Use the web_search tool to find information. Be thorough and systematic in your research approach."""
    
    return Agent(
        name="IterativeSearchAgent",
        instructions=instructions,
        tools=[WebSearchTool()]
    )


async def demonstrate_iterative_search():
    """
    Demonstrate the iterative search agent with comprehensive callbacks.
    
    This function shows how the callback system enables sophisticated
    agent behaviors that would be difficult to implement with traditional
    fixed execution patterns.
    """
    print("🚀 ITERATIVE SEARCH AGENT DEMONSTRATION")
    print("="*60)
    print("This example showcases the ADK callback system by implementing")
    print("a sophisticated iterative search agent that:")
    print("• Searches for information across multiple iterations")
    print("• Dynamically refines queries based on previous results")
    print("• Accumulates and manages context intelligently")
    print("• Detects loops to prevent repetitive searches")
    print("• Determines synthesis completion automatically")
    print("• Provides detailed progress tracking and statistics")
    print("="*60)
    
    # Create the agent
    agent = await create_iterative_search_agent()
    
    # Create the callback system
    callbacks = IterativeSearchCallbacks(
        max_iterations=4,
        synthesis_threshold=4
    )
    
    # Configure the runner
    config = RunnerConfig(
        agent=agent,
        session_provider=None,  # Mock session provider
        max_llm_calls=6,
        callbacks=callbacks,
        enable_context_accumulation=True,
        enable_loop_detection=True
    )
    
    # Test queries to demonstrate different behaviors
    test_queries = [
        "What are the main applications of machine learning in industry?",
        "How is artificial intelligence being used in healthcare?",
        "What are the latest trends in AI and machine learning?"
    ]
    
    class MockModelProvider:
        async def get_completion(self, state, agent, config):
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'search_1',
                            'type': 'function',
                            'function': {
                                'name': 'web_search',
                                'arguments': json.dumps({'query': state.messages[-1].content})
                            }
                        }
                    ]
                }
            }
    model_provider = MockModelProvider()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*20} TEST QUERY {i} {'='*20}")
        
        message = Message(role='user', content=query)
        context = {
            'user_id': 'demo_user',
            'session_id': f'demo_session_{i}',
            'query_id': f'query_{i}'
        }
        
        try:
            # Run the agent with full callback instrumentation
            result = await run_agent(
                config=config,
                message=message,
                context=context,
                model_provider=model_provider
            )
            
            print(f"\n📋 FINAL RESULT:")
            print(f"   Query: {query}")
            print(f"   Response: {result.content.content[:200]}...")
            print(f"   Execution time: {result.execution_time_ms:.0f}ms")
            print(f"   Metadata: {result.metadata}")
            assert result.content.content is not None
            assert result.execution_time_ms > 0
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        # Reset callback state for next query
        callbacks = IterativeSearchCallbacks(
            max_iterations=4,
            synthesis_threshold=4
        )
        config = RunnerConfig(
            agent=agent,
            session_provider=None,
            max_llm_calls=6,
            callbacks=callbacks,
            enable_context_accumulation=True,
            enable_loop_detection=True
        )
    
    print(f"\n{'='*60}")
    print("🎉 DEMONSTRATION COMPLETE")
    print("The callback system enabled sophisticated iterative behavior")
    print("that would be impossible with traditional fixed execution!")
    print("="*60)


async def demonstrate_callback_features():
    """
    Demonstrate specific callback features in isolation.
    
    This shows how individual callbacks can be used for specific
    customizations without implementing the full iterative pattern.
    """
    print("\n🔧 INDIVIDUAL CALLBACK FEATURES DEMONSTRATION")
    print("="*55)
    
    class FeatureDemoCallbacks:
        """Demonstrate individual callback capabilities."""
        
        async def on_start(self, context, message, session_state):
            print(f"🎬 on_start: Processing query '{message.content}'")
            print(f"   Context: {context}")
        
        async def on_before_llm_call(self, agent, message, session_state):
            print(f"🤖 on_before_llm_call: Intercepting LLM call")
            print(f"   Original message: {message.content}")
            
            # Demonstrate message modification
            if "simple" in message.content.lower():
                modified_message = Message(
                    role='user', 
                    content=f"Please provide a detailed explanation about: {message.content}"
                )
                print(f"   Modified to: {modified_message.content}")
                return {'message': modified_message}
            return None
        
        async def on_after_llm_call(self, response, session_state):
            print(f"✅ on_after_llm_call: LLM responded with {len(response.content)} characters")
            
            # Demonstrate response modification
            if len(response.content) < 50:
                enhanced_response = Message(
                    role='assistant',
                    content=f"{response.content}\n\n[Enhanced by callback: This response was automatically expanded for completeness.]"
                )
                print(f"   Enhanced short response")
                return enhanced_response
            return None
        
        async def on_iteration_start(self, iteration):
            print(f"🔄 on_iteration_start: Beginning iteration {iteration}")
            if iteration > 2:
                print(f"   Limiting to 2 iterations for demo")
                return {'continue_iteration': False}
            return None
        
        async def on_complete(self, response):
            print(f"🏁 on_complete: Execution finished successfully")
            print(f"   Final response length: {len(response.content.content)} characters")
            print(f"   Metadata: {response.metadata}")
    
    # Create simple agent for feature demo
    def simple_instructions(state):
        return "You are a helpful assistant that provides clear, concise answers."
    
    agent = Agent(
        name="FeatureDemoAgent",
        instructions=simple_instructions,
        tools=[]
    )
    
    callbacks = FeatureDemoCallbacks()
    config = RunnerConfig(
        agent=agent,
        session_provider=None,
        callbacks=callbacks,
        max_llm_calls=3
    )
    
    # Test different scenarios
    test_cases = [
        "What is AI?",  # Simple query to trigger enhancement
        "Tell me about machine learning algorithms in detail"  # Complex query
    ]
    
    for query in test_cases:
        print(f"\n--- Testing: '{query}' ---")
        message = Message(role='user', content=query)
        
        try:
            result = await run_agent(
                config=config,
                message=message,
                context={'demo': True}
            )
            print(f"Result: {result.content.content[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


# ========== Main Execution ==========

async def main():
    """Run the complete iterative search agent demonstration."""
    try:
        # Demonstrate the full iterative search capabilities
        await demonstrate_iterative_search()
        
        # Demonstrate individual callback features
        await demonstrate_callback_features()
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 JAF ADK - Iterative Search Agent Example")
    print("Demonstrating the power of the comprehensive callback system")
    print("="*60)
    
    # Run the demonstration
    asyncio.run(main())
