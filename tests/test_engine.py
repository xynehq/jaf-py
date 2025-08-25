"""
Tests for the JAF engine module.

Based on the TypeScript engine.test.ts file from the original implementation.
"""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from jaf.core.engine import run
from jaf.core.types import (
    Agent,
    AgentNotFound,
    CompletedOutcome,
    ContentRole,
    ErrorOutcome,
    InvalidValidationResult,
    MaxTurnsExceeded,
    Message,
    ModelBehaviorError,
    ModelConfig,
    RunConfig,
    RunState,
    Tool,
    ToolSchema,
    ValidationResult,
    ValidValidationResult,
    create_run_id,
    create_trace_id,
)


class MockModelProvider:
    """Mock model provider for testing."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self.responses = responses
        self.call_count = 0

    async def get_completion(self, state, agent, config) -> Dict[str, Any]:
        """Return a mock completion response."""
        if self.call_count >= len(self.responses):
            response = {
                'message': {
                    'content': 'Default response',
                    'tool_calls': None
                }
            }
        else:
            response = self.responses[self.call_count]

        self.call_count += 1
        return response


class SimpleToolArgs(BaseModel):
    """Simple tool arguments for testing."""
    message: str


class SimpleTool:
    """Simple tool for testing."""

    def __init__(self, name: str = "simple_tool", result: str = "tool result"):
        self.name = name
        self.result = result
        self._schema = ToolSchema(
            name=name,
            description="A simple test tool",
            parameters=SimpleToolArgs
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, args: SimpleToolArgs, context: Any) -> str:
        return f"{self.result}: {args.message}"


class HandoffToolArgs(BaseModel):
    """Handoff tool arguments."""
    target_agent: str
    reason: str


class HandoffTool:
    """Tool that performs agent handoffs."""

    def __init__(self):
        self._schema = ToolSchema(
            name="handoff_to_agent",
            description="Handoff to another agent",
            parameters=HandoffToolArgs
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def execute(self, args: HandoffToolArgs, context: Any) -> str:
        return f'{{"handoff_to": "{args.target_agent}", "reason": "{args.reason}"}}'


def create_test_agent(
    name: str = "test_agent",
    tools: Optional[List[Tool]] = None,
    instructions: Optional[str] = None
) -> Agent:
    """Create a test agent."""

    def agent_instructions(state) -> str:
        return instructions or f"You are {name}, a helpful assistant."

    return Agent(
        name=name,
        instructions=agent_instructions,
        tools=tools,
        output_codec=None,
        handoffs=None,
        model_config=ModelConfig(name="gpt-4o", temperature=0.7)
    )


def create_test_run_state(
    messages: Optional[List[Message]] = None,
    context: Optional[Dict[str, Any]] = None,
    agent_name: str = "test_agent"
) -> RunState:
    """Create a test run state."""
    return RunState(
        run_id=create_run_id("test-run"),
        trace_id=create_trace_id("test-trace"),
        messages=messages or [Message(role=ContentRole.USER, content='Hello')],
        current_agent_name=agent_name,
        context=context or {},
        turn_count=0
    )


@pytest.mark.asyncio
async def test_simple_completion():
    """Test simple text completion without tools."""
    # Setup
    agent = create_test_agent()
    model_provider = MockModelProvider([
        {
            'message': {
                'content': 'Hello! How can I help you?',
                'tool_calls': None
            }
        }
    ])

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, CompletedOutcome)
    assert result.outcome.output == 'Hello! How can I help you?'
    assert result.final_state.turn_count == 1
    assert len(result.final_state.messages) == 2  # User + Assistant


@pytest.mark.asyncio
async def test_tool_call_execution():
    """Test agent with tool calling."""
    # Setup
    tool = SimpleTool("test_tool", "Success")
    agent = create_test_agent(tools=[tool])

    model_provider = MockModelProvider([
        {
            'message': {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'test_tool',
                            'arguments': '{"message": "test input"}'
                        }
                    }
                ]
            }
        },
        {
            'message': {
                'content': 'The tool returned: Success: test input',
                'tool_calls': None
            }
        }
    ])

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, CompletedOutcome)
    assert "Success: test input" in result.outcome.output
    assert len(result.final_state.messages) == 4  # User + Assistant + Tool + Assistant


@pytest.mark.asyncio
async def test_agent_handoff():
    """Test agent handoff functionality."""
    # Setup
    handoff_tool = HandoffTool()
    source_agent = create_test_agent("source_agent", tools=[handoff_tool])
    source_agent = Agent(
        name="source_agent",
        instructions=source_agent.instructions,
        tools=[handoff_tool],
        output_codec=None,
        handoffs=["target_agent"],  # Allow handoff to target_agent
        model_config=source_agent.model_config
    )

    target_agent = create_test_agent("target_agent")

    model_provider = MockModelProvider([
        # Source agent calls handoff tool
        {
            'message': {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'handoff_to_agent',
                            'arguments': '{"target_agent": "target_agent", "reason": "Specialized task"}'
                        }
                    }
                ]
            }
        },
        # Target agent responds
        {
            'message': {
                'content': 'I am the target agent, ready to help!',
                'tool_calls': None
            }
        }
    ])

    state = create_test_run_state(agent_name="source_agent")
    config = RunConfig(
        agent_registry={
            "source_agent": source_agent,
            "target_agent": target_agent
        },
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, CompletedOutcome)
    assert result.final_state.current_agent_name == "target_agent"
    assert "target agent" in result.outcome.output


@pytest.mark.asyncio
async def test_max_turns_exceeded():
    """Test max turns limit."""
    # Setup
    agent = create_test_agent()
    model_provider = MockModelProvider([
        {
            'message': {
                'content': 'Response',
                'tool_calls': [
                    {
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'non_existent_tool',
                            'arguments': '{}'
                        }
                    }
                ]
            }
        }
    ] * 10)  # Repeat to exceed max turns

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=3  # Low limit to trigger error
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, ErrorOutcome)
    assert isinstance(result.outcome.error, MaxTurnsExceeded)
    assert result.outcome.error.turns >= 3


@pytest.mark.asyncio
async def test_agent_not_found():
    """Test error when agent is not found."""
    # Setup
    agent = create_test_agent()
    model_provider = MockModelProvider([])

    state = create_test_run_state(agent_name="non_existent_agent")
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, ErrorOutcome)
    assert isinstance(result.outcome.error, AgentNotFound)
    assert result.outcome.error.agent_name == "non_existent_agent"


@pytest.mark.asyncio
async def test_tool_not_found():
    """Test error when tool is not found."""
    # Setup
    agent = create_test_agent()  # No tools
    model_provider = MockModelProvider([
        {
            'message': {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'non_existent_tool',
                            'arguments': '{}'
                        }
                    }
                ]
            }
        },
        {
            'message': {
                'content': 'Continuing after tool error',
                'tool_calls': None
            }
        }
    ])

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, CompletedOutcome)
    # Should have tool error message in conversation
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    assert len(tool_messages) == 1
    assert 'tool_not_found' in tool_messages[0].content


@pytest.mark.asyncio
async def test_input_guardrails():
    """Test input guardrail validation."""
    # Setup
    async def bad_word_guardrail(input_text: str) -> ValidationResult:
        if "badword" in input_text.lower():
            return InvalidValidationResult(error_message="Contains inappropriate content")
        return ValidValidationResult()

    agent = create_test_agent()
    model_provider = MockModelProvider([])

    state = create_test_run_state(messages=[
        Message(role=ContentRole.USER, content='This contains badword in it')
    ])
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10,
        initial_input_guardrails=[bad_word_guardrail]
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, ErrorOutcome)
    assert "inappropriate content" in str(result.outcome.error)


@pytest.mark.asyncio
async def test_event_tracing():
    """Test that events are properly emitted during execution."""
    # Setup
    events = []

    def event_collector(event):
        events.append(event)

    agent = create_test_agent()
    model_provider = MockModelProvider([
        {
            'message': {
                'content': 'Hello!',
                'tool_calls': None
            }
        }
    ])

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10,
        on_event=event_collector
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert len(events) >= 3  # run_start, llm_call_start, llm_call_end, run_end
    assert events[0].type == 'run_start'
    assert events[-1].type == 'run_end'

    # Check for LLM events
    llm_start_events = [e for e in events if e.type == 'llm_call_start']
    llm_end_events = [e for e in events if e.type == 'llm_call_end']
    assert len(llm_start_events) == 1
    assert len(llm_end_events) == 1


@pytest.mark.asyncio
async def test_model_behavior_error():
    """Test handling of model behavior errors."""
    # Setup
    agent = create_test_agent()

    # Mock provider that raises an exception
    class FailingModelProvider:
        async def get_completion(self, state, agent, config):
            raise Exception("Model service unavailable")

    model_provider = FailingModelProvider()

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    result = await run(state, config)

    # Assert
    assert isinstance(result.outcome, ErrorOutcome)
    assert isinstance(result.outcome.error, ModelBehaviorError)
    assert "Model service unavailable" in result.outcome.error.detail


@pytest.mark.asyncio
async def test_streaming_tool_call_execution():
    """Test agent with tool calling in a streaming context."""
    # Setup
    from jaf.core.streaming import run_streaming, StreamingEventType, StreamingCollector

    tool = SimpleTool("streaming_test_tool", "Streamed Success")
    agent = create_test_agent(tools=[tool])

    model_provider = MockModelProvider([
        {
            'message': {
                'content': '',
                'tool_calls': [
                    {
                        'id': 'call_stream_123',
                        'type': 'function',
                        'function': {
                            'name': 'streaming_test_tool',
                            'arguments': '{"message": "streamed input"}'
                        }
                    }
                ]
            }
        },
        {
            'message': {
                'content': 'The streamed tool returned: Streamed Success: streamed input',
                'tool_calls': None
            }
        }
    ])

    state = create_test_run_state()
    config = RunConfig(
        agent_registry={"test_agent": agent},
        model_provider=model_provider,
        max_turns=10
    )

    # Execute
    stream = run_streaming(state, config)
    collector = StreamingCollector()
    final_buffer = await collector.collect_stream(stream)

    # Assert
    assert final_buffer.is_complete
    assert final_buffer.error is None
    
    # Check for tool call and result events
    tool_call_events = [e for e in collector.events if e.type == StreamingEventType.TOOL_CALL]
    tool_result_events = [e for e in collector.events if e.type == StreamingEventType.TOOL_RESULT]
    
    assert len(tool_call_events) == 1
    assert tool_call_events[0].data.tool_name == "streaming_test_tool"
    assert tool_call_events[0].data.arguments == {"message": "streamed input"}
    call_id = tool_call_events[0].data.call_id

    assert len(tool_result_events) == 1
    assert tool_result_events[0].data.tool_name == "streaming_test_tool"
    assert "Streamed Success: streamed input" in tool_result_events[0].data.result
    assert tool_result_events[0].data.call_id == call_id

    # Check final content
    final_message = final_buffer.get_final_message()
    assert "Streamed Success: streamed input" in final_message.content
