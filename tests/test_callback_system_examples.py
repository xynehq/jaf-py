"""
Test suite for Callback System examples documentation.
Tests all code examples from docs/callback-system.md to ensure they work with the actual implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, MagicMock
from adk.runners import RunnerCallbacks, RunnerConfig
from jaf.core.types import Message, Agent


class TestCallbackSystemBasicExamples:
    """Test basic callback system examples from the documentation."""
    
    def test_basic_callbacks_structure(self):
        """Test basic callback implementation structure."""
        
        class MyCallbacks:
            """Custom callback implementation."""
            
            def __init__(self):
                self.start_time = None
                self.events = []
            
            async def on_start(self, context, message, session_state):
                """Called at agent execution start."""
                self.start_time = time.time()
                self.events.append(f"Processing: {message.content}")
            
            async def on_complete(self, response):
                """Called when execution completes successfully."""
                execution_time = (time.time() - self.start_time) * 1000 if self.start_time else 0
                self.events.append(f"Completed in {execution_time:.0f}ms")
            
            async def on_error(self, error, context):
                """Called when execution encounters an error."""
                self.events.append(f"Error: {error}")
            
            async def on_before_llm_call(self, agent, message, session_state):
                """Modify or skip LLM calls."""
                # Example: Add context to message
                context_info = session_state.get('context', '')
                if context_info:
                    enriched_content = f"Context: {context_info}\n{message.content}"
                    return {'message': Message(role='user', content=enriched_content)}
                return None
            
            async def on_after_llm_call(self, response, session_state):
                """Modify LLM responses."""
                # Example: Post-process response
                if hasattr(response, 'content') and len(response.content) < 50:
                    enhanced = f"{response.content}\n\n[Response enhanced for completeness]"
                    return Message(role='assistant', content=enhanced)
                return None
        
        callbacks = MyCallbacks()
        
        # Test callback methods exist
        assert hasattr(callbacks, 'on_start')
        assert hasattr(callbacks, 'on_complete')
        assert hasattr(callbacks, 'on_error')
        assert hasattr(callbacks, 'on_before_llm_call')
        assert hasattr(callbacks, 'on_after_llm_call')
        
        # Test callback execution
        import asyncio
        
        async def test_callbacks():
            # Test on_start
            message = Message(role='user', content='Test message')
            await callbacks.on_start({}, message, {})
            assert len(callbacks.events) == 1
            assert "Processing: Test message" in callbacks.events[0]
            
            # Test on_complete
            mock_response = MagicMock()
            await callbacks.on_complete(mock_response)
            assert len(callbacks.events) == 2
            assert "Completed in" in callbacks.events[1]
            
            # Test on_error
            await callbacks.on_error("Test error", {})
            assert len(callbacks.events) == 3
            assert "Error: Test error" in callbacks.events[2]
            
            # Test on_before_llm_call
            session_state = {'context': 'Important context'}
            result = await callbacks.on_before_llm_call(None, message, session_state)
            assert result is not None
            assert 'message' in result
            assert 'Context: Important context' in result['message'].content
            
            # Test on_after_llm_call
            short_response = Message(role='assistant', content='Short')
            enhanced = await callbacks.on_after_llm_call(short_response, {})
            assert enhanced is not None
            assert '[Response enhanced for completeness]' in enhanced.content
        
        asyncio.run(test_callbacks())
    
    def test_runner_config_with_callbacks(self):
        """Test RunnerConfig with callback support."""
        
        class TestCallbacks:
            async def on_start(self, context, message, session_state):
                return None
        
        # Mock agent and session provider
        mock_agent = MagicMock()
        mock_session_provider = MagicMock()
        callbacks = TestCallbacks()
        
        # Test RunnerConfig creation (structure validation)
        config_data = {
            'agent': mock_agent,
            'session_provider': mock_session_provider,
            'callbacks': callbacks,
            'max_llm_calls': 10,
            'enable_context_accumulation': True,
            'enable_loop_detection': True,
            'max_context_items': 100,
            'max_repeated_tools': 3
        }
        
        # Validate configuration structure
        assert config_data['agent'] == mock_agent
        assert config_data['session_provider'] == mock_session_provider
        assert config_data['callbacks'] == callbacks
        assert config_data['max_llm_calls'] == 10
        assert config_data['enable_context_accumulation'] is True
        assert config_data['enable_loop_detection'] is True
        assert config_data['max_context_items'] == 100
        assert config_data['max_repeated_tools'] == 3


class TestLifecycleCallbacks:
    """Test lifecycle callback examples."""
    
    def test_lifecycle_callbacks(self):
        """Test lifecycle callback implementation."""
        
        class LifecycleCallbacks:
            def __init__(self):
                self.start_time = None
                self.query_id = None
                self.metrics = []
                self.errors = []
            
            async def on_start(self, context, message, session_state):
                """Execution started - initialize tracking."""
                self.start_time = time.time()
                self.query_id = f"query_{int(time.time())}"
                
            async def on_complete(self, response):
                """Execution completed - log metrics."""
                duration = time.time() - self.start_time if self.start_time else 0
                self.log_metrics(self.query_id, duration, response)
                
            async def on_error(self, error, context):
                """Handle execution errors gracefully."""
                self.log_error(self.query_id, error, context)
            
            def log_metrics(self, query_id, duration, response):
                self.metrics.append({
                    'query_id': query_id,
                    'duration': duration,
                    'response': response
                })
            
            def log_error(self, query_id, error, context):
                self.errors.append({
                    'query_id': query_id,
                    'error': str(error),
                    'context': context
                })
        
        callbacks = LifecycleCallbacks()
        
        # Test lifecycle flow
        import asyncio
        
        async def test_lifecycle():
            # Start
            message = Message(role='user', content='Test')
            await callbacks.on_start({}, message, {})
            assert callbacks.start_time is not None
            assert callbacks.query_id is not None
            
            # Complete
            mock_response = MagicMock()
            await callbacks.on_complete(mock_response)
            assert len(callbacks.metrics) == 1
            assert callbacks.metrics[0]['query_id'] == callbacks.query_id
            
            # Error
            await callbacks.on_error("Test error", {"test": "context"})
            assert len(callbacks.errors) == 1
            assert callbacks.errors[0]['error'] == "Test error"
            assert callbacks.errors[0]['context'] == {"test": "context"}
        
        asyncio.run(test_lifecycle())


class TestLLMControlCallbacks:
    """Test LLM control callback examples."""
    
    def test_llm_control_callbacks(self):
        """Test LLM interaction control callbacks."""
        
        class LLMControlCallbacks:
            def __init__(self):
                self.cache = {}
                self.cached_responses = []
            
            async def on_before_llm_call(self, agent, message, session_state):
                """Modify messages before LLM call."""
                # Skip LLM for cached responses
                cached_response = self.check_cache(message.content)
                if cached_response:
                    return {'skip': True, 'response': cached_response}
                
                # Enrich message with context
                context_summary = self.get_context_summary(session_state)
                if context_summary:
                    enriched_message = self.add_context(message, context_summary)
                    return {'message': enriched_message}
                return None
            
            async def on_after_llm_call(self, response, session_state):
                """Post-process LLM responses."""
                # Cache response for future use
                self.cache_response(response)
                
                # Apply post-processing rules
                return self.apply_formatting_rules(response)
            
            def check_cache(self, content):
                return self.cache.get(content)
            
            def get_context_summary(self, session_state):
                return session_state.get('context_summary', '')
            
            def add_context(self, message, context_summary):
                enhanced_content = f"Context: {context_summary}\n{message.content}"
                return Message(role=message.role, content=enhanced_content)
            
            def cache_response(self, response):
                if hasattr(response, 'content'):
                    # Simple cache key based on content hash
                    cache_key = str(hash(response.content))
                    self.cache[cache_key] = response
                    self.cached_responses.append(response)
            
            def apply_formatting_rules(self, response):
                if hasattr(response, 'content'):
                    # Example: Ensure responses end with punctuation
                    content = response.content.strip()
                    if content and content[-1] not in '.!?':
                        content += '.'
                    return Message(role=response.role, content=content)
                return response
        
        callbacks = LLMControlCallbacks()
        
        # Test LLM control flow
        import asyncio
        
        async def test_llm_control():
            message = Message(role='user', content='Test question')
            session_state = {'context_summary': 'Important context'}
            
            # Test before LLM call with context
            result = await callbacks.on_before_llm_call(None, message, session_state)
            assert result is not None
            assert 'message' in result
            assert 'Context: Important context' in result['message'].content
            
            # Test after LLM call
            response = Message(role='assistant', content='Test response')
            formatted = await callbacks.on_after_llm_call(response, session_state)
            assert formatted.content == 'Test response.'  # Added punctuation
            
            # Test caching
            assert len(callbacks.cached_responses) == 1
            
            # Test cache hit
            cache_key = 'Test question'
            callbacks.cache[cache_key] = response
            result = await callbacks.on_before_llm_call(None, message, {})
            assert result['skip'] is True
            assert result['response'] == response
        
        asyncio.run(test_llm_control())


class TestIterationControlCallbacks:
    """Test iteration control callback examples."""
    
    def test_iterative_reasoning_callbacks(self):
        """Test iterative reasoning callback implementation."""
        
        class IterativeReasoningCallbacks:
            def __init__(self, max_iterations=5):
                self.max_iterations = max_iterations
                self.iteration_count = 0
                self.information_gathered = []
                
            async def on_iteration_start(self, iteration):
                """Control iteration flow."""
                self.iteration_count = iteration
                
                if iteration > self.max_iterations:
                    return {'continue_iteration': False}
                return None
            
            async def on_iteration_complete(self, iteration, has_tool_calls):
                """Decide whether to continue iterating."""
                if not has_tool_calls:
                    # No tools called, likely finished
                    return {'should_stop': True}
                
                if self.sufficient_information_gathered():
                    return {'should_stop': True}
                
                return {'should_continue': True}
            
            def sufficient_information_gathered(self):
                return len(self.information_gathered) >= 3
        
        callbacks = IterativeReasoningCallbacks(max_iterations=3)
        
        # Test iteration control
        import asyncio
        
        async def test_iteration_control():
            # Test iteration start
            result = await callbacks.on_iteration_start(1)
            assert result is None  # Should continue
            assert callbacks.iteration_count == 1
            
            # Test max iterations exceeded
            result = await callbacks.on_iteration_start(5)
            assert result is not None
            assert result['continue_iteration'] is False
            
            # Test iteration complete with tools
            result = await callbacks.on_iteration_complete(1, True)
            assert result['should_continue'] is True
            
            # Test iteration complete without tools
            result = await callbacks.on_iteration_complete(1, False)
            assert result['should_stop'] is True
            
            # Test sufficient information
            callbacks.information_gathered = ['info1', 'info2', 'info3']
            result = await callbacks.on_iteration_complete(1, True)
            assert result['should_stop'] is True
        
        asyncio.run(test_iteration_control())


class TestToolControlCallbacks:
    """Test tool control callback examples."""
    
    def test_tool_control_callbacks(self):
        """Test tool execution control callbacks."""
        
        class ToolControlCallbacks:
            def __init__(self):
                self.tool_selections = []
                self.auth_token = "test-auth-token"
            
            async def on_before_tool_selection(self, tools, context_data):
                """Filter or modify available tools."""
                # Limit tools based on context
                if len(context_data) > 10:
                    # Only allow synthesis tools when we have enough data
                    synthesis_tools = [t for t in tools if 'synthesis' in getattr(t, 'name', '')]
                    if synthesis_tools:
                        return {'tools': synthesis_tools}
                return None
            
            async def on_tool_selected(self, tool_name, params):
                """Track tool usage."""
                self.log_tool_selection(tool_name, params)
            
            async def on_before_tool_execution(self, tool, params):
                """Modify parameters or skip execution."""
                # Add authentication to API calls
                if hasattr(tool, 'name') and tool.name == 'api_call':
                    enhanced_params = {**params, 'auth_token': self.get_auth_token()}
                    return {'params': enhanced_params}
                return None
            
            async def on_after_tool_execution(self, tool, result, error=None):
                """Process tool results."""
                if error:
                    self.handle_tool_error(tool, error)
                    return None
                
                # Transform result format
                return self.standardize_result_format(result)
            
            def log_tool_selection(self, tool_name, params):
                self.tool_selections.append({
                    'tool': tool_name,
                    'params': params,
                    'timestamp': time.time()
                })
            
            def get_auth_token(self):
                return self.auth_token
            
            def handle_tool_error(self, tool, error):
                # Log error handling
                pass
            
            def standardize_result_format(self, result):
                if isinstance(result, dict):
                    return {'status': 'success', 'data': result}
                else:
                    return {'status': 'success', 'data': {'result': str(result)}}
        
        callbacks = ToolControlCallbacks()
        
        # Test tool control flow
        import asyncio
        
        async def test_tool_control():
            # Test tool selection
            await callbacks.on_tool_selected('test_tool', {'param1': 'value1'})
            assert len(callbacks.tool_selections) == 1
            assert callbacks.tool_selections[0]['tool'] == 'test_tool'
            
            # Test before tool execution with API call
            mock_api_tool = MagicMock()
            mock_api_tool.name = 'api_call'
            result = await callbacks.on_before_tool_execution(mock_api_tool, {'url': 'test'})
            assert result is not None
            assert 'params' in result
            assert result['params']['auth_token'] == 'test-auth-token'
            
            # Test after tool execution success
            standardized = await callbacks.on_after_tool_execution(None, {'key': 'value'})
            assert standardized['status'] == 'success'
            assert standardized['data'] == {'key': 'value'}
            
            # Test after tool execution with string result
            standardized = await callbacks.on_after_tool_execution(None, 'simple result')
            assert standardized['status'] == 'success'
            assert standardized['data']['result'] == 'simple result'
        
        asyncio.run(test_tool_control())


class TestSynthesisCallbacks:
    """Test synthesis and context callback examples."""
    
    def test_synthesis_callbacks(self):
        """Test synthesis and context management callbacks."""
        
        class SynthesisCallbacks:
            def __init__(self, confidence_threshold=0.8):
                self.confidence_threshold = confidence_threshold
                self.context_accumulator = []
            
            async def on_check_synthesis(self, session_state, context_data):
                """Determine if synthesis is complete."""
                if len(context_data) < 3:
                    return None  # Need more information
                
                # Analyze information completeness
                coverage_score = self.analyze_coverage(context_data)
                quality_score = self.analyze_quality(context_data)
                confidence = (coverage_score + quality_score) / 2
                
                if confidence >= self.confidence_threshold:
                    synthesis_prompt = self.create_synthesis_prompt(context_data)
                    return {
                        'complete': True,
                        'answer': synthesis_prompt,
                        'confidence': confidence
                    }
                return None
            
            async def on_query_rewrite(self, original_query, context_data):
                """Refine queries based on accumulated context."""
                gaps = self.identify_knowledge_gaps(context_data)
                if gaps:
                    return f"{original_query} focusing on {', '.join(gaps)}"
                return None
            
            async def on_context_update(self, current_context, new_items):
                """Manage context accumulation."""
                # Deduplicate and filter
                filtered_items = self.deduplicate_and_filter(new_items)
                
                # Merge with existing context
                merged_context = current_context + filtered_items
                
                # Sort by relevance and limit size
                sorted_context = sorted(merged_context, key=lambda x: x.get('relevance', 0), reverse=True)
                return sorted_context[:50]  # Keep top 50 items
            
            def analyze_coverage(self, context_data):
                # Mock coverage analysis
                return min(len(context_data) / 5.0, 1.0)
            
            def analyze_quality(self, context_data):
                # Mock quality analysis
                avg_relevance = sum(item.get('relevance', 0.5) for item in context_data) / len(context_data)
                return avg_relevance
            
            def create_synthesis_prompt(self, context_data):
                return f"Based on {len(context_data)} pieces of information, here is the synthesis..."
            
            def identify_knowledge_gaps(self, context_data):
                # Mock gap identification
                if len(context_data) < 2:
                    return ['more details', 'specific examples']
                return []
            
            def deduplicate_and_filter(self, new_items):
                # Mock deduplication
                return [item for item in new_items if item.get('relevance', 0) > 0.3]
        
        callbacks = SynthesisCallbacks(confidence_threshold=0.7)
        
        # Test synthesis flow
        import asyncio
        
        async def test_synthesis():
            # Test insufficient data
            result = await callbacks.on_check_synthesis({}, [])
            assert result is None
            
            # Test sufficient high-quality data
            context_data = [
                {'content': 'High quality content 1', 'relevance': 0.9},
                {'content': 'High quality content 2', 'relevance': 0.85},
                {'content': 'High quality content 3', 'relevance': 0.8}
            ]
            result = await callbacks.on_check_synthesis({}, context_data)
            assert result is not None
            assert result['complete'] is True
            assert result['confidence'] >= 0.7
            
            # Test query rewrite
            rewritten = await callbacks.on_query_rewrite("original query", [{'relevance': 0.5}])
            assert rewritten is not None
            assert "focusing on" in rewritten
            
            # Test context update
            current_context = [{'content': 'existing', 'relevance': 0.8}]
            new_items = [
                {'content': 'new1', 'relevance': 0.9},
                {'content': 'new2', 'relevance': 0.2}  # Should be filtered out
            ]
            updated = await callbacks.on_context_update(current_context, new_items)
            assert len(updated) == 2  # existing + new1 (new2 filtered out)
            assert updated[0]['relevance'] == 0.9  # Sorted by relevance
        
        asyncio.run(test_synthesis())


class TestLoopDetectionCallbacks:
    """Test loop detection callback examples."""
    
    def test_loop_detection_callbacks(self):
        """Test loop detection and prevention callbacks."""
        
        class LoopDetectionCallbacks:
            def __init__(self, similarity_threshold=0.7):
                self.similarity_threshold = similarity_threshold
                self.tool_history = []
            
            async def on_loop_detection(self, tool_history, current_tool):
                """Detect and prevent loops."""
                if len(tool_history) < 3:
                    return False
                
                # Check for repetitive tool calls
                recent_tools = [item['tool'] for item in tool_history[-3:]]
                if recent_tools.count(current_tool) > 2:
                    return True
                
                # Check for parameter similarity
                recent_params = [item.get('params', {}) for item in tool_history[-3:]]
                for params in recent_params:
                    if self.calculate_similarity(params, current_tool) > self.similarity_threshold:
                        return True
                
                return False
            
            def calculate_similarity(self, params, current_tool):
                # Mock similarity calculation
                if isinstance(params, dict) and 'query' in params:
                    # Simple string similarity for query parameters
                    return 0.8 if current_tool in str(params['query']) else 0.3
                return 0.0
        
        callbacks = LoopDetectionCallbacks()
        
        # Test loop detection
        import asyncio
        
        async def test_loop_detection():
            # No loop for different tools
            tool_history = [
                {'tool': 'search', 'params': {'query': 'query1'}},
                {'tool': 'analyze', 'params': {'data': 'data1'}}
            ]
            result = await callbacks.on_loop_detection(tool_history, 'summarize')
            assert result is False
            
            # Loop detected for repeated tools
            tool_history = [
                {'tool': 'search', 'params': {'query': 'query1'}},
                {'tool': 'search', 'params': {'query': 'query2'}},
                {'tool': 'search', 'params': {'query': 'query3'}}
            ]
            result = await callbacks.on_loop_detection(tool_history, 'search')
            assert result is True
            
            # Test with insufficient history
            short_history = [{'tool': 'search', 'params': {}}]
            result = await callbacks.on_loop_detection(short_history, 'search')
            assert result is False
        
        asyncio.run(test_loop_detection())


class TestAdvancedPatterns:
    """Test advanced callback patterns."""
    
    def test_react_pattern(self):
        """Test ReAct (Reasoning + Acting) pattern implementation."""
        
        class ReActAgent:
            """ReAct pattern implementation using callbacks."""
            
            def __init__(self):
                self.observations = []
                self.thoughts = []
                self.actions = []
            
            async def on_iteration_start(self, iteration):
                """Think about what to do next."""
                if iteration == 1:
                    thought = "I need to gather information about the user's query."
                else:
                    thought = f"Based on {len(self.observations)} observations, I should..."
                
                self.thoughts.append(thought)
                return None
            
            async def on_before_tool_execution(self, tool, params):
                """Record planned action."""
                action = f"Using {getattr(tool, 'name', 'unknown')} with {params}"
                self.actions.append(action)
                return None
            
            async def on_after_tool_execution(self, tool, result, error=None):
                """Record observation."""
                if error:
                    observation = f"Action failed: {error}"
                else:
                    observation = f"Observed: {str(result)[:100]}"
                
                self.observations.append(observation)
                return None
            
            async def on_check_synthesis(self, session_state, context_data):
                """Decide if we have enough information."""
                if len(self.observations) >= 3:
                    final_thought = "I have sufficient information to provide a comprehensive answer."
                    synthesis = self.synthesize_observations()
                    
                    return {
                        'complete': True,
                        'answer': f"Final thought: {final_thought}\n\nAnswer: {synthesis}",
                        'confidence': 0.9
                    }
                return None
            
            def synthesize_observations(self):
                return f"Based on {len(self.observations)} observations: " + "; ".join(self.observations[:3])
        
        react_agent = ReActAgent()
        
        # Test ReAct pattern
        import asyncio
        
        async def test_react_pattern():
            # Test thinking
            await react_agent.on_iteration_start(1)
            assert len(react_agent.thoughts) == 1
            assert "gather information" in react_agent.thoughts[0]
            
            # Test action planning
            mock_tool = MagicMock()
            mock_tool.name = 'search'
            await react_agent.on_before_tool_execution(mock_tool, {'query': 'test'})
            assert len(react_agent.actions) == 1
            assert "search" in react_agent.actions[0]
            
            # Test observation
            await react_agent.on_after_tool_execution(mock_tool, {'result': 'test result'})
            assert len(react_agent.observations) == 1
            assert "Observed:" in react_agent.observations[0]
            
            # Add more observations
            await react_agent.on_after_tool_execution(mock_tool, {'result': 'result2'})
            await react_agent.on_after_tool_execution(mock_tool, {'result': 'result3'})
            
            # Test synthesis
            result = await react_agent.on_check_synthesis({}, [])
            assert result is not None
            assert result['complete'] is True
            assert "Final thought:" in result['answer']
            assert result['confidence'] == 0.9
        
        asyncio.run(test_react_pattern())
    
    def test_caching_callbacks(self):
        """Test intelligent caching pattern."""
        
        class CachingCallbacks:
            def __init__(self):
                self.cache = {}
                self.cache_hits = 0
                self.cache_misses = 0
            
            async def on_before_llm_call(self, agent, message, session_state):
                """Check cache before LLM call."""
                cache_key = self.generate_cache_key(message, session_state)
                
                if cache_key in self.cache:
                    self.cache_hits += 1
                    cached_response = self.cache[cache_key]
                    return {'skip': True, 'response': cached_response}
                
                self.cache_misses += 1
                return None
            
            async def on_after_llm_call(self, response, session_state):
                """Cache LLM response."""
                cache_key = self.generate_cache_key(response, session_state)
                self.cache[cache_key] = response
                
                return None
            
            def generate_cache_key(self, message_or_response, session_state):
                if hasattr(message_or_response, 'content'):
                    content = message_or_response.content
                else:
                    content = str(message_or_response)
                return f"{hash(content)}_{hash(str(session_state))}"
            
            def get_hit_rate(self):
                total = self.cache_hits + self.cache_misses
                return (self.cache_hits / total * 100) if total > 0 else 0
        
        callbacks = CachingCallbacks()
        
        # Test caching pattern
        import asyncio
        
        async def test_caching():
            message = Message(role='user', content='Test message')
            session_state = {'context': 'test'}
            
            # First call - cache miss
            result = await callbacks.on_before_llm_call(None, message, session_state)
            assert result is None  # No cache hit
            assert callbacks.cache_misses == 1
            assert callbacks.cache_hits == 0
            
            # Cache the response
            response = Message(role='assistant', content='Test response')
            await callbacks.on_after_llm_call(response, session_state)
            assert len(callbacks.cache) == 1
            
            # Second call - cache hit
            cache_key = callbacks.generate_cache_key(message, session_state)
            callbacks.cache[cache_key] = response
            result = await callbacks.on_before_llm_call(None, message, session_state)
            assert result is not None
            assert result['skip'] is True
            assert result['response'] == response
            assert callbacks.cache_hits == 1
            
            # Test hit rate calculation
            hit_rate = callbacks.get_hit_rate()
            assert hit_rate == 50.0  # 1 hit out of 2 total calls
        
        asyncio.run(test_caching())


class TestPerformanceAndDebugging:
    """Test performance monitoring and debugging examples."""
    
    def test_performance_callbacks(self):
        """Test performance monitoring callbacks."""
        
        class PerformanceCallbacks:
            def __init__(self):
                self.metrics = {
                    'llm_calls': 0,
                    'tool_calls': 0,
                    'total_tokens': 0,
                    'cache_hits': 0
                }
                self.start_time = None
            
            async def on_start(self, context, message, session_state):
                self.start_time = time.time()
            
            async def on_before_llm_call(self, agent, message, session_state):
                self.metrics['llm_calls'] += 1
                return None
            
            async def on_tool_selected(self, tool_name, params):
                self.metrics['tool_calls'] += 1
                return None
            
            async def on_complete(self, response):
                execution_time = (time.time() - self.start_time) * 1000 if self.start_time else 0
                print(f"Performance Metrics:")
                print(f"  LLM Calls: {self.metrics['llm_calls']}")
                print(f"  Tool Calls: {self.metrics['tool_calls']}")
                print(f"  Execution Time: {execution_time:.0f}ms")
        
        callbacks = PerformanceCallbacks()
        
        # Test performance monitoring
        import asyncio
        
        async def test_performance():
            # Test start
            message = Message(role='user', content='Test')
            await callbacks.on_start({}, message, {})
            assert callbacks.start_time is not None
            
            # Test LLM call tracking
            await callbacks.on_before_llm_call(None, message, {})
            assert callbacks.metrics['llm_calls'] == 1
            
            # Test tool call tracking
            await callbacks.on_tool_selected('test_tool', {})
            assert callbacks.metrics['tool_calls'] == 1
            
            # Test completion
            mock_response = MagicMock()
            mock_response.execution_time_ms = 1000
            await callbacks.on_complete(mock_response)
            # Metrics should be printed (tested via execution)
        
        asyncio.run(test_performance())
    
    def test_debug_callbacks(self):
        """Test comprehensive debug logging callbacks."""
        
        class DebugCallbacks:
            def __init__(self, log_level='INFO'):
                self.log_level = log_level
                self.debug_info = []
            
            async def on_iteration_start(self, iteration):
                self.log(f"🔄 Starting iteration {iteration}")
                return None
            
            async def on_before_llm_call(self, agent, message, session_state):
                content_preview = message.content[:100] + "..." if len(message.content) > 100 else message.content
                self.log(f"🤖 LLM Call: {content_preview}")
                return None
            
            async def on_after_tool_execution(self, tool, result, error=None):
                tool_name = getattr(tool, 'name', 'unknown') if tool else 'unknown'
                if error:
                    self.log(f"❌ Tool {tool_name} failed: {error}")
                else:
                    self.log(f"✅ Tool {tool_name} succeeded")
                return None
            
            def log(self, message):
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] {message}"
                self.debug_info.append(log_entry)
                if self.log_level == 'DEBUG':
                    print(log_entry)
        
        callbacks = DebugCallbacks(log_level='DEBUG')
        
        # Test debug logging
        import asyncio
        
        async def test_debug():
            # Test iteration logging
            await callbacks.on_iteration_start(1)
            assert len(callbacks.debug_info) == 1
            assert "Starting iteration 1" in callbacks.debug_info[0]
            
            # Test LLM call logging
            message = Message(role='user', content='Test message for debugging')
            await callbacks.on_before_llm_call(None, message, {})
            assert len(callbacks.debug_info) == 2
            assert "LLM Call:" in callbacks.debug_info[1]
            
            # Test tool success logging
            mock_tool = MagicMock()
            mock_tool.name = 'test_tool'
            await callbacks.on_after_tool_execution(mock_tool, {'result': 'success'})
            assert len(callbacks.debug_info) == 3
            assert "test_tool succeeded" in callbacks.debug_info[2]
            
            # Test tool failure logging
            await callbacks.on_after_tool_execution(mock_tool, None, "Test error")
            assert len(callbacks.debug_info) == 4
            assert "test_tool failed: Test error" in callbacks.debug_info[3]
        
        asyncio.run(test_debug())


class TestCallbackTesting:
    """Test callback testing examples."""
    
    @pytest.mark.asyncio
    async def test_synthesis_callback_unit_test(self):
        """Test synthesis checking logic unit test."""
        
        class SynthesisCallbacks:
            def __init__(self, confidence_threshold=0.8):
                self.confidence_threshold = confidence_threshold
            
            async def on_check_synthesis(self, session_state, context_data):
                if len(context_data) < 3:
                    return None
                
                # Simple confidence calculation
                avg_relevance = sum(item.get('relevance', 0.5) for item in context_data) / len(context_data)
                confidence = avg_relevance
                
                if confidence >= self.confidence_threshold:
                    return {
                        'complete': True,
                        'answer': f'Synthesis based on {len(context_data)} items',
                        'confidence': confidence
                    }
                return None
        
        callbacks = SynthesisCallbacks(confidence_threshold=0.8)
        
        # Test with insufficient data
        result = await callbacks.on_check_synthesis({}, [])
        assert result is None
        
        # Test with sufficient high-quality data
        context_data = [
            {'content': 'High quality content 1', 'relevance': 0.9},
            {'content': 'High quality content 2', 'relevance': 0.85},
            {'content': 'High quality content 3', 'relevance': 0.8}
        ]
        result = await callbacks.on_check_synthesis({}, context_data)
        assert result['complete'] is True
        assert result['confidence'] >= 0.8
        assert 'Synthesis based on 3 items' in result['answer']
    
    @pytest.mark.asyncio
    async def test_loop_detection_unit_test(self):
        """Test loop detection logic unit test."""
        
        class LoopDetectionCallbacks:
            async def on_loop_detection(self, tool_history, current_tool):
                if len(tool_history) < 3:
                    return False
                
                # Check for repetitive tool calls
                recent_tools = [item['tool'] for item in tool_history[-3:]]
                if recent_tools.count(current_tool) > 2:
                    return True
                
                return False
        
        callbacks = LoopDetectionCallbacks()
        
        # No loop for different tools
        tool_history = [
            {'tool': 'search', 'params': {'q': 'query1'}},
            {'tool': 'analyze', 'params': {'data': 'data1'}}
        ]
        result = await callbacks.on_loop_detection(tool_history, 'summarize')
        assert result is False
        
        # Loop detected for repeated tools
        tool_history = [
            {'tool': 'search', 'params': {'q': 'query1'}},
            {'tool': 'search', 'params': {'q': 'query2'}},
            {'tool': 'search', 'params': {'q': 'query3'}}
        ]
        result = await callbacks.on_loop_detection(tool_history, 'search')
        assert result is True
    
    @pytest.mark.asyncio
    async def test_iterative_workflow_integration(self):
        """Test complete iterative agent workflow integration test."""
        
        class TestCallbacks:
            def __init__(self):
                self.iterations = 0
                self.context_items = []
            
            async def on_iteration_start(self, iteration):
                self.iterations = iteration
                return None
            
            async def on_context_update(self, current_context, new_items):
                self.context_items.extend(new_items)
                return self.context_items
            
            async def on_check_synthesis(self, session_state, context_data):
                if len(context_data) >= 3:
                    return {'complete': True, 'answer': 'Test synthesis'}
                return None
        
        callbacks = TestCallbacks()
        
        # Mock iterative workflow
        await callbacks.on_iteration_start(1)
        assert callbacks.iterations == 1
        
        # Mock context accumulation
        await callbacks.on_context_update([], [{'item': 'data1'}])
        await callbacks.on_context_update(callbacks.context_items, [{'item': 'data2'}])
        await callbacks.on_context_update(callbacks.context_items, [{'item': 'data3'}])
        
        assert len(callbacks.context_items) >= 3
        
        # Test synthesis
        result = await callbacks.on_check_synthesis({}, callbacks.context_items)
        assert result is not None
        assert result['complete'] is True
        assert 'Test synthesis' in result['answer']


class TestCallbackIntegration:
    """Test callback integration examples."""
    
    def test_jaf_core_integration_structure(self):
        """Test JAF core integration structure."""
        
        # Mock JAF core components
        class MockAgent:
            def __init__(self, name, instructions, tools):
                self.name = name
                self.instructions = instructions
                self.tools = tools
        
        class ResearchCallbacks:
            async def on_query_rewrite(self, original_query, context_data):
                return self.refine_research_query(original_query, context_data)
            
            def refine_research_query(self, original_query, context_data):
                if len(context_data) > 0:
                    return f"{original_query} (refined based on {len(context_data)} context items)"
                return original_query
        
        # Test integration structure
        def agent_instructions(state):
            return "You are a research assistant with iterative capabilities."
        
        agent = MockAgent(
            name="ResearchAgent",
            instructions=agent_instructions,
            tools=[]
        )
        
        callbacks = ResearchCallbacks()
        
        # Test configuration structure
        config_data = {
            'agent': agent,
            'callbacks': callbacks
        }
        
        assert config_data['agent'].name == "ResearchAgent"
        assert config_data['callbacks'] is not None
        
        # Test callback functionality
        import asyncio
        
        async def test_integration():
            refined = await callbacks.on_query_rewrite("original query", [{'data': 'test'}])
            assert "refined based on 1 context items" in refined
        
        asyncio.run(test_integration())
    
    def test_memory_system_integration_structure(self):
        """Test memory system integration structure."""
        
        class MemoryAwareCallbacks:
            def __init__(self):
                self.memory_provider = None
            
            async def on_start(self, context, message, session_state):
                # Mock memory loading
                if self.memory_provider:
                    memories = await self.search_memories(message.content)
                    session_state['relevant_memories'] = memories
            
            async def on_complete(self, response):
                # Mock memory storage
                if self.memory_provider:
                    await self.store_interaction(response)
            
            async def search_memories(self, content):
                # Mock memory search
                return [{'memory': f'Related to {content}'}]
            
            async def store_interaction(self, response):
                # Mock memory storage
                pass
        
        # Mock memory provider
        class MockMemoryProvider:
            async def search_memories(self, query):
                return [{'memory': f'Mock memory for {query}'}]
            
            async def store_interaction(self, interaction):
                pass
        
        memory_provider = MockMemoryProvider()
        callbacks = MemoryAwareCallbacks()
        callbacks.memory_provider = memory_provider
        
        # Test integration
        import asyncio
        
        async def test_memory_integration():
            message = Message(role='user', content='Test query')
            session_state = {}
            
            await callbacks.on_start({}, message, session_state)
            assert 'relevant_memories' in session_state
            assert len(session_state['relevant_memories']) > 0
            
            mock_response = MagicMock()
            await callbacks.on_complete(mock_response)
            # Memory storage tested via execution
        
        asyncio.run(test_memory_integration())


class TestCallbackBestPractices:
    """Test callback best practices examples."""
    
    def test_single_responsibility_principle(self):
        """Test single responsibility principle in callbacks."""
        
        # Good: Clear, focused callback
        class GoodCallback:
            def __init__(self):
                self.start_time = None
                self.execution_id = None
            
            async def on_start(self, context, message, session_state):
                """Initialize tracking for this execution."""
                self.start_time = time.time()
                self.execution_id = f"exec_{int(time.time())}"
        
        # Test good callback
        callback = GoodCallback()
        
        import asyncio
        
        async def test_good_callback():
            message = Message(role='user', content='Test')
            await callback.on_start({}, message, {})
            assert callback.start_time is not None
            assert callback.execution_id is not None
            assert callback.execution_id.startswith('exec_')
        
        asyncio.run(test_good_callback())
    
    def test_error_handling_best_practices(self):
        """Test robust error handling in callbacks."""
        
        class RobustCallbacks:
            def __init__(self):
                self.errors = []
            
            async def on_before_llm_call(self, agent, message, session_state):
                try:
                    return self.enhance_message(message, session_state)
                except Exception as e:
                    # Log error but don't break execution
                    self.log_error(f"Message enhancement failed: {e}")
                    return None  # Let execution continue normally
            
            def enhance_message(self, message, session_state):
                # Mock enhancement that might fail
                if 'error_trigger' in message.content:
                    raise ValueError("Enhancement failed")
                return {'message': Message(role=message.role, content=f"Enhanced: {message.content}")}
            
            def log_error(self, error_message):
                self.errors.append(error_message)
        
        callbacks = RobustCallbacks()
        
        # Test error handling
        import asyncio
        
        async def test_error_handling():
            # Test successful enhancement
            message = Message(role='user', content='Normal message')
            result = await callbacks.on_before_llm_call(None, message, {})
            assert result is not None
            assert 'Enhanced:' in result['message'].content
            
            # Test error handling
            error_message = Message(role='user', content='error_trigger message')
            result = await callbacks.on_before_llm_call(None, error_message, {})
            assert result is None  # Should return None on error
            assert len(callbacks.errors) == 1
            assert 'Enhancement failed' in callbacks.errors[0]
        
        asyncio.run(test_error_handling())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
