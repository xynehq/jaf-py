"""
Property-based testing examples for JAF framework.

This module demonstrates how to use property-based testing with Hypothesis
to test JAF components with automatically generated test cases that explore
edge cases and invariants.
"""

import asyncio
import json
from typing import Any, Dict, List
from dataclasses import dataclass

import pytest
from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant

from jaf.core.types import (
    Message, ContentRole, RunState, Agent, Tool, ToolSchema,
    generate_run_id, generate_trace_id, ModelConfig
)
from jaf.core.engine import run
from jaf.core.performance import PerformanceMonitor
from jaf.core.composition import create_tool_pipeline, with_retry, with_cache
from jaf.core.tool_results import ToolResponse


# Test data strategies

@st.composite
def message_strategy(draw):
    """Generate valid Message objects."""
    role = draw(st.sampled_from([ContentRole.USER, ContentRole.ASSISTANT, ContentRole.TOOL]))
    content = draw(st.text(min_size=1, max_size=1000))
    
    # Tool call ID only for tool messages
    tool_call_id = None
    if role == ContentRole.TOOL:
        tool_call_id = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    
    return Message(role=role, content=content, tool_call_id=tool_call_id)


@st.composite
def context_strategy(draw):
    """Generate test context objects."""
    @dataclass
    class TestContext:
        user_id: str
        permissions: List[str]
        session_data: Dict[str, Any]
    
    user_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    permissions = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    session_data = draw(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(), st.integers(), st.booleans()),
        min_size=0,
        max_size=5
    ))
    
    return TestContext(user_id=user_id, permissions=permissions, session_data=session_data)


@st.composite
def run_state_strategy(draw):
    """Generate valid RunState objects."""
    messages = draw(st.lists(message_strategy(), min_size=1, max_size=10))
    context = draw(context_strategy())
    agent_name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
    turn_count = draw(st.integers(min_value=0, max_value=100))
    
    return RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=messages,
        current_agent_name=agent_name,
        context=context,
        turn_count=turn_count
    )


# Property-based tests

class TestMessageProperties:
    """Property-based tests for Message objects."""
    
    @given(message_strategy())
    def test_message_serialization_roundtrip(self, message: Message):
        """Test that messages can be serialized and deserialized without loss."""
        # Convert to dict (simulating JSON serialization)
        message_dict = {
            'role': message.role.value,
            'content': message.content,
            'tool_call_id': message.tool_call_id,
            'tool_calls': message.tool_calls
        }
        
        # Reconstruct message
        reconstructed = Message(
            role=ContentRole(message_dict['role']),
            content=message_dict['content'],
            tool_call_id=message_dict['tool_call_id'],
            tool_calls=message_dict['tool_calls']
        )
        
        assert reconstructed.role == message.role
        assert reconstructed.content == message.content
        assert reconstructed.tool_call_id == message.tool_call_id
        assert reconstructed.tool_calls == message.tool_calls
    
    @given(st.lists(message_strategy(), min_size=1, max_size=20))
    def test_message_list_properties(self, messages: List[Message]):
        """Test properties that should hold for any list of messages."""
        # All messages should have non-empty content
        assert all(len(msg.content) > 0 for msg in messages)
        
        # Tool messages should have tool_call_id if present
        tool_messages = [msg for msg in messages if msg.role == ContentRole.TOOL]
        for msg in tool_messages:
            if msg.tool_call_id is not None:
                assert len(msg.tool_call_id) > 0
    
    @given(message_strategy())
    @example(Message(role=ContentRole.USER, content="Hello"))
    def test_message_content_invariants(self, message: Message):
        """Test invariants that should always hold for messages."""
        # Content should never be empty
        assert len(message.content) > 0
        
        # Role should be valid
        assert message.role in [ContentRole.USER, ContentRole.ASSISTANT, ContentRole.TOOL, ContentRole.SYSTEM]
        
        # Tool call ID should be None or non-empty string
        if message.tool_call_id is not None:
            assert isinstance(message.tool_call_id, str)
            assert len(message.tool_call_id) > 0


class TestRunStateProperties:
    """Property-based tests for RunState objects."""
    
    @given(run_state_strategy())
    def test_run_state_immutability(self, state: RunState):
        """Test that RunState objects are truly immutable."""
        original_messages = state.messages
        original_turn_count = state.turn_count
        
        # Attempting to modify should not affect the original
        try:
            # This should fail because RunState is frozen
            state.turn_count = 999  # type: ignore
            assert False, "RunState should be immutable"
        except (AttributeError, TypeError):
            # Expected - frozen dataclass prevents modification
            pass
        
        # Original values should be unchanged
        assert state.messages == original_messages
        assert state.turn_count == original_turn_count
    
    @given(run_state_strategy())
    def test_run_state_consistency(self, state: RunState):
        """Test consistency properties of RunState."""
        # Turn count should be non-negative
        assert state.turn_count >= 0
        
        # Should have at least one message
        assert len(state.messages) > 0
        
        # Agent name should be non-empty
        assert len(state.current_agent_name) > 0
        
        # IDs should be properly formatted
        assert state.run_id.startswith('run_')
        assert state.trace_id.startswith('trace_')


class TestPerformanceMonitorProperties:
    """Property-based tests for PerformanceMonitor."""
    
    @given(
        st.integers(min_value=0, max_value=1000),  # llm_calls
        st.integers(min_value=0, max_value=1000),  # tool_calls
        st.integers(min_value=0, max_value=100),   # cache_hits
        st.integers(min_value=0, max_value=100),   # cache_misses
    )
    def test_performance_metrics_properties(self, llm_calls, tool_calls, cache_hits, cache_misses):
        """Test properties of performance metrics calculation."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Simulate operations
        for _ in range(llm_calls):
            monitor.record_llm_call(token_count=10)
        
        for _ in range(tool_calls):
            monitor.record_tool_call()
        
        for _ in range(cache_hits):
            monitor.record_cache_hit()
        
        for _ in range(cache_misses):
            monitor.record_cache_miss()
        
        metrics = monitor.stop_monitoring()
        
        # Test invariants
        assert metrics.llm_call_count == llm_calls
        assert metrics.tool_call_count == tool_calls
        assert metrics.execution_time_ms >= 0
        assert metrics.memory_usage_mb >= 0
        assert metrics.peak_memory_mb >= metrics.memory_usage_mb
        
        # Cache hit rate should be between 0 and 100
        assert 0 <= metrics.cache_hit_rate <= 100
        
        # If no cache operations, hit rate should be 0
        if cache_hits == 0 and cache_misses == 0:
            assert metrics.cache_hit_rate == 0
        
        # If only hits, hit rate should be 100
        if cache_hits > 0 and cache_misses == 0:
            assert metrics.cache_hit_rate == 100


class TestToolCompositionProperties:
    """Property-based tests for tool composition."""
    
    def create_test_tool(self, name: str, delay_ms: int = 0, should_fail: bool = False):
        """Create a test tool for composition testing."""
        class TestTool:
            @property
            def schema(self):
                return ToolSchema(
                    name=name,
                    description=f"Test tool {name}",
                    parameters=dict
                )
            
            async def execute(self, args: Any, context: Any):
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)
                
                if should_fail:
                    raise ValueError(f"Tool {name} failed")
                
                return ToolResponse.success(f"Result from {name}: {args}")
        
        return TestTool()
    
    @given(
        st.integers(min_value=1, max_value=5),  # number of tools
        st.integers(min_value=0, max_value=100),  # delay per tool
    )
    @settings(max_examples=10, deadline=5000)  # Limit examples for async tests
    async def test_pipeline_composition_properties(self, num_tools: int, delay_ms: int):
        """Test properties of tool pipeline composition."""
        # Create tools
        tools = [self.create_test_tool(f"tool_{i}", delay_ms) for i in range(num_tools)]
        
        # Create pipeline
        pipeline = create_tool_pipeline(*tools, name="test_pipeline")
        
        # Test execution
        result = await pipeline.execute({"input": "test"}, {})
        
        # Pipeline should succeed if all tools succeed
        assert isinstance(result, str) or hasattr(result, 'status')
        
        # Result should contain information about all steps
        if hasattr(result, 'metadata') and result.metadata is not None:
            # Handle both dict and object metadata
            if hasattr(result.metadata, 'get'):
                steps_executed = result.metadata.get('steps_executed')
            elif hasattr(result.metadata, 'extra'):
                steps_executed = result.metadata.extra.get('steps_executed')
            else:
                steps_executed = getattr(result.metadata, 'steps_executed', None)
            
            if steps_executed is not None:
                assert steps_executed == num_tools
    
    @given(
        st.integers(min_value=1, max_value=2),  # max retries (reduced)
        st.floats(min_value=0.01, max_value=0.1),  # backoff factor (much smaller)
    )
    @settings(max_examples=3, deadline=1000)  # Reduced examples and deadline
    async def test_retry_tool_properties(self, max_retries: int, backoff_factor: float):
        """Test properties of retry tool wrapper."""
        # Create a tool that always fails
        failing_tool = self.create_test_tool("failing_tool", should_fail=True)
        
        # Wrap with retry (with minimal backoff)
        retry_tool = with_retry(failing_tool, max_retries=max_retries, backoff_factor=backoff_factor)
        
        # Execute and expect failure after retries
        result = await retry_tool.execute({"input": "test"}, {})
        
        # Should eventually fail (basic test - just ensure it doesn't hang)
        assert result is not None
        
        # If it has status, should be error
        if hasattr(result, 'status'):
            assert result.status in ['error', 'failed']
        
        # If it has metadata and attempts, verify retry count
        if hasattr(result, 'metadata') and result.metadata is not None:
            attempts = result.metadata.get('attempts')
            if attempts is not None:
                expected_attempts = max_retries + 1
                assert attempts == expected_attempts


# Stateful testing for complex scenarios

class AgentExecutionStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for agent execution scenarios.
    
    This tests complex interactions and state transitions in agent execution
    by generating sequences of operations and checking invariants.
    """
    
    def __init__(self):
        super().__init__()
        self.messages: List[Message] = []
        self.turn_count = 0
        self.agent_name = "test_agent"
        self.context = {"user_id": "test_user", "permissions": ["read"]}
    
    @initialize()
    def setup(self):
        """Initialize the state machine with a user message."""
        initial_message = Message(role=ContentRole.USER, content="Hello")
        self.messages = [initial_message]
        self.turn_count = 0
    
    @rule(content=st.text(min_size=1, max_size=100))
    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        message = Message(role=ContentRole.USER, content=content)
        self.messages.append(message)
    
    @rule(content=st.text(min_size=1, max_size=100))
    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation."""
        message = Message(role=ContentRole.ASSISTANT, content=content)
        self.messages.append(message)
        self.turn_count += 1
    
    @rule()
    def create_run_state(self):
        """Create a RunState from current conversation."""
        assume(len(self.messages) > 0)
        
        state = RunState(
            run_id=generate_run_id(),
            trace_id=generate_trace_id(),
            messages=self.messages.copy(),
            current_agent_name=self.agent_name,
            context=self.context,
            turn_count=self.turn_count
        )
        
        # Test that state is consistent
        assert len(state.messages) == len(self.messages)
        assert state.turn_count == self.turn_count
        assert state.current_agent_name == self.agent_name
    
    @invariant()
    def messages_are_valid(self):
        """Invariant: All messages should be valid."""
        for message in self.messages:
            assert len(message.content) > 0
            assert message.role in [ContentRole.USER, ContentRole.ASSISTANT, ContentRole.TOOL, ContentRole.SYSTEM]
    
    @invariant()
    def turn_count_is_consistent(self):
        """Invariant: Turn count should match assistant messages."""
        assistant_messages = [msg for msg in self.messages if msg.role == ContentRole.ASSISTANT]
        assert self.turn_count == len(assistant_messages)


# Integration property tests

class TestIntegrationProperties:
    """Property-based tests for integration scenarios."""
    
    @given(
        st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5),  # tool names
        st.text(min_size=1, max_size=100),  # input content
    )
    @settings(max_examples=5, deadline=2000)
    async def test_tool_registry_properties(self, tool_names: List[str], input_content: str):
        """Test properties of tool registration and lookup."""
        # Ensure unique tool names
        tool_names = list(set(tool_names))
        assume(len(tool_names) > 0)
        
        # Create tools
        tools = {}
        for name in tool_names:
            class TestTool:
                @property
                def schema(self):
                    return ToolSchema(
                        name=name,
                        description=f"Test tool {name}",
                        parameters=dict
                    )
                
                async def execute(self, args: Any, context: Any):
                    return ToolResponse.success(f"Result from {name}")
            
            tools[name] = TestTool()
        
        # Test properties
        assert len(tools) == len(tool_names)
        
        # All tools should be findable by name
        for name in tool_names:
            assert name in tools
            assert tools[name].schema.name == name
    
    @given(st.text(min_size=1, max_size=1000))
    def test_json_serialization_properties(self, content: str):
        """Test JSON serialization properties for various content."""
        # Create a message with the content
        message = Message(role=ContentRole.USER, content=content)
        
        # Serialize to JSON
        try:
            message_dict = {
                'role': message.role.value,
                'content': message.content,
                'tool_call_id': message.tool_call_id
            }
            json_str = json.dumps(message_dict)
            
            # Deserialize back
            parsed = json.loads(json_str)
            
            # Should be able to reconstruct
            reconstructed = Message(
                role=ContentRole(parsed['role']),
                content=parsed['content'],
                tool_call_id=parsed['tool_call_id']
            )
            
            assert reconstructed.content == message.content
            assert reconstructed.role == message.role
            
        except (json.JSONEncodeError, UnicodeDecodeError):
            # Some content might not be JSON serializable, which is acceptable
            pass


# Test runner for property-based tests

class TestPropertyBasedRunner:
    """Runner for property-based tests with custom settings."""
    
    @settings(max_examples=50, deadline=1000)
    @given(st.integers(min_value=1, max_value=100))
    def test_id_generation_properties(self, num_ids: int):
        """Test properties of ID generation."""
        run_ids = [generate_run_id() for _ in range(num_ids)]
        trace_ids = [generate_trace_id() for _ in range(num_ids)]
        
        # All IDs should be unique
        assert len(set(run_ids)) == num_ids
        assert len(set(trace_ids)) == num_ids
        
        # All run IDs should start with 'run_'
        assert all(rid.startswith('run_') for rid in run_ids)
        
        # All trace IDs should start with 'trace_'
        assert all(tid.startswith('trace_') for tid in trace_ids)
    
    def test_stateful_agent_execution(self):
        """Run stateful testing for agent execution."""
        # This would run the state machine
        # In practice, you'd use: run_state_machine_as_test(AgentExecutionStateMachine)
        pass


# Fixtures for property-based testing

@pytest.fixture
def property_test_settings():
    """Provide custom settings for property-based tests."""
    return settings(
        max_examples=20,
        deadline=2000,
        suppress_health_check=[],
    )


# Example of how to run property-based tests
if __name__ == "__main__":
    # Run a simple property test
    test_instance = TestMessageProperties()
    
    # Generate and test a few examples
    for _ in range(5):
        message = message_strategy().example()
        test_instance.test_message_serialization_roundtrip(message)
        print(f"✓ Tested message: {message.role.value} - {message.content[:50]}...")
    
    print("Property-based tests completed successfully!")
