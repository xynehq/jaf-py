"""
Tests for A2A agent functionality

Tests agent creation, transformation, and execution functions
for the A2A implementation.
"""

from datetime import datetime

import pytest

from jaf.a2a.agent import (
    add_artifact_to_a2a_task,
    add_message_to_state,
    complete_a2a_task,
    create_a2a_agent,
    create_a2a_data_message,
    create_a2a_task,
    create_a2a_text_message,
    create_a2a_tool,
    create_initial_agent_state,
    create_run_config_for_a2a_agent,
    create_user_message,
    execute_a2a_agent,
    execute_a2a_agent_with_streaming,
    extract_text_from_a2a_message,
    process_agent_query,
    transform_a2a_agent_to_jaf,
    transform_a2a_tool_to_jaf,
    transform_to_run_state,
    update_a2a_task_status,
    update_state_from_run_result,
)
from jaf.a2a.types import (
    A2AAgent,
    A2AAgentTool,
    AgentState,
    StreamEvent,
    create_a2a_message,
    create_a2a_text_part,
)
from jaf.core.types import Agent, Message, RunConfig


# Mock functions for testing
async def mock_tool_function(args, context):
    """Mock tool function for testing"""
    return {"result": f"Tool executed with args: {args}"}


def mock_instructions(state):
    """Mock instructions function"""
    return "You are a test agent."


class MockModelProvider:
    """Mock model provider for testing"""

    async def get_completion(self, state, agent, config):
        return {"message": {"content": "Mock response from agent", "tool_calls": None}}


class TestAgentCreation:
    """Test A2A agent creation functions"""

    def test_create_a2a_agent_basic(self):
        """Test basic A2A agent creation"""
        agent = create_a2a_agent("TestAgent", "A test agent", "You are helpful", [])

        assert isinstance(agent, A2AAgent)
        assert agent.name == "TestAgent"
        assert agent.description == "A test agent"
        assert agent.instruction == "You are helpful"
        assert len(agent.tools) == 0
        assert "text/plain" in agent.supported_content_types
        assert "application/json" in agent.supported_content_types

    def test_create_a2a_agent_with_tools(self):
        """Test A2A agent creation with tools"""
        tool = create_a2a_tool("test_tool", "A test tool", {"type": "object"}, mock_tool_function)

        agent = create_a2a_agent(
            "ToolAgent", "Agent with tools", "Use tools wisely", [tool], ["text/plain", "text/html"]
        )

        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"
        assert agent.supported_content_types == ["text/plain", "text/html"]

    def test_create_a2a_tool(self):
        """Test A2A tool creation"""
        parameters = {"type": "object", "properties": {"message": {"type": "string"}}}

        tool = create_a2a_tool("echo_tool", "Echoes back the input", parameters, mock_tool_function)

        assert isinstance(tool, A2AAgentTool)
        assert tool.name == "echo_tool"
        assert tool.description == "Echoes back the input"
        assert tool.parameters == parameters
        assert tool.execute == mock_tool_function


class TestAgentTransformation:
    """Test A2A to JAF agent transformation"""

    def test_transform_a2a_agent_to_jaf(self):
        """Test transforming A2A agent to JAF agent"""
        a2a_tool = create_a2a_tool("test_tool", "Test tool", {"type": "object"}, mock_tool_function)

        a2a_agent = create_a2a_agent("TestAgent", "Test agent", "You are a test agent", [a2a_tool])

        jaf_agent = transform_a2a_agent_to_jaf(a2a_agent)

        assert isinstance(jaf_agent, Agent)
        assert jaf_agent.name == "TestAgent"
        assert len(jaf_agent.tools) == 1

        # Test instructions function
        instructions = jaf_agent.instructions(None)
        assert instructions == "You are a test agent"

    @pytest.mark.asyncio
    async def test_transform_a2a_tool_to_jaf(self):
        """Test transforming A2A tool to JAF tool"""
        a2a_tool = create_a2a_tool(
            "math_tool", "Performs math calculations", {"type": "object"}, mock_tool_function
        )

        jaf_tool = transform_a2a_tool_to_jaf(a2a_tool)

        # Check that it has the required Tool protocol attributes
        assert hasattr(jaf_tool, "schema")
        assert hasattr(jaf_tool, "execute")
        assert jaf_tool.schema.name == "math_tool"
        assert jaf_tool.schema.description == "Performs math calculations"

        # Test tool execution
        result = await jaf_tool.execute("test_args", {"context": "data"})
        assert "Tool executed with args: test_args" in str(result)

    @pytest.mark.asyncio
    async def test_tool_result_handling(self):
        """Test tool result format handling"""

        # Tool that returns ToolResult format
        async def tool_with_result_format(args, context):
            return {"result": "formatted result"}

        a2a_tool = create_a2a_tool(
            "format_tool", "Tool with result format", {}, tool_with_result_format
        )

        jaf_tool = transform_a2a_tool_to_jaf(a2a_tool)
        result = await jaf_tool.execute("test", {})

        assert result == "formatted result"


class TestStateManagement:
    """Test agent state management functions"""

    def test_create_initial_agent_state(self):
        """Test creating initial agent state"""
        state = create_initial_agent_state("session_123")

        assert isinstance(state, AgentState)
        assert state.sessionId == "session_123"
        assert len(state.messages) == 0
        assert state.context == {}
        assert len(state.artifacts) == 0
        assert state.timestamp is not None

    def test_add_message_to_state(self):
        """Test adding message to agent state"""
        initial_state = create_initial_agent_state("session_123")
        message = create_a2a_message(
            role="user", parts=[create_a2a_text_part("Hello")], context_id="session_123"
        )

        new_state = add_message_to_state(initial_state, message)

        # Original state unchanged (immutable)
        assert len(initial_state.messages) == 0

        # New state has message
        assert len(new_state.messages) == 1
        assert new_state.messages[0] == message
        assert new_state.sessionId == "session_123"

    def test_update_state_from_run_result(self):
        """Test updating state from run result"""
        initial_state = create_initial_agent_state("session_123")

        # Mock run result with artifacts
        class MockOutcome:
            def __init__(self):
                self.artifacts = ["artifact1", "artifact2"]

        outcome = MockOutcome()

        updated_state = update_state_from_run_result(initial_state, outcome)

        assert len(updated_state.artifacts) == 2
        assert updated_state.artifacts == ["artifact1", "artifact2"]
        assert updated_state.sessionId == "session_123"

    def test_create_user_message(self):
        """Test creating user message"""
        message = create_user_message("Hello, agent!")

        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.content == "Hello, agent!"


class TestMessageProcessing:
    """Test message processing functions"""

    def test_extract_text_from_a2a_message(self):
        """Test extracting text from A2A message"""
        # Message with text parts
        message = {
            "parts": [
                {"kind": "text", "text": "First part"},
                {"kind": "data", "data": {"key": "value"}},
                {"kind": "text", "text": "Second part"},
            ]
        }

        text = extract_text_from_a2a_message(message)
        assert text == "First part\nSecond part"

        # Empty message
        empty_text = extract_text_from_a2a_message({})
        assert empty_text == ""

        # Message with no parts
        no_parts = extract_text_from_a2a_message({"parts": []})
        assert no_parts == ""

    def test_create_a2a_text_message(self):
        """Test creating A2A text message"""
        message = create_a2a_text_message("Hello world", "ctx_123", "task_456")

        assert message["role"] == "agent"
        assert message["contextId"] == "ctx_123"
        assert message["taskId"] == "task_456"
        assert message["kind"] == "message"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["kind"] == "text"
        assert message["parts"][0]["text"] == "Hello world"
        assert message["messageId"].startswith("msg_")

    def test_create_a2a_data_message(self):
        """Test creating A2A data message"""
        data = {"result": "success", "count": 42}
        message = create_a2a_data_message(data, "ctx_789")

        assert message["role"] == "agent"
        assert message["contextId"] == "ctx_789"
        assert message["kind"] == "message"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["kind"] == "data"
        assert message["parts"][0]["data"] == data


class TestTaskManagement:
    """Test A2A task management functions"""

    def test_create_a2a_task(self):
        """Test creating A2A task"""
        message = create_a2a_message(
            role="user", parts=[create_a2a_text_part("Task request")], context_id="ctx_123"
        )

        task = create_a2a_task(message.model_dump(), "ctx_123")

        assert task["id"].startswith("task_")
        assert task["contextId"] == "ctx_123"
        assert task["status"]["state"] == "submitted"
        assert task["kind"] == "task"
        assert len(task["history"]) == 1
        assert len(task["artifacts"]) == 0

    def test_update_a2a_task_status(self):
        """Test updating A2A task status"""
        message = create_a2a_message(
            role="user", parts=[create_a2a_text_part("Test")], context_id="ctx_123"
        )

        task = create_a2a_task(message.model_dump(), "ctx_123")

        # Update to working status
        working_task = update_a2a_task_status(task, "working")

        # Original task unchanged
        assert task["status"]["state"] == "submitted"

        # New task updated
        assert working_task["status"]["state"] == "working"
        assert working_task["id"] == task["id"]

    def test_add_artifact_to_a2a_task(self):
        """Test adding artifact to A2A task"""
        message = create_a2a_message(
            role="user", parts=[create_a2a_text_part("Test")], context_id="ctx_123"
        )

        task = create_a2a_task(message.model_dump(), "ctx_123")

        parts = [{"kind": "text", "text": "Artifact content"}]
        task_with_artifact = add_artifact_to_a2a_task(task, parts, "test_artifact")

        # Original task unchanged
        assert len(task["artifacts"]) == 0

        # New task has artifact
        assert len(task_with_artifact["artifacts"]) == 1
        artifact = task_with_artifact["artifacts"][0]
        assert artifact["name"] == "test_artifact"
        assert artifact["parts"] == parts
        assert artifact["artifactId"].startswith("artifact_")

    def test_complete_a2a_task(self):
        """Test completing A2A task"""
        message = create_a2a_message(
            role="user", parts=[create_a2a_text_part("Test")], context_id="ctx_123"
        )

        task = create_a2a_task(message.model_dump(), "ctx_123")

        completed_task = complete_a2a_task(task, "Task completed successfully")

        assert completed_task["status"]["state"] == "completed"
        assert len(completed_task["artifacts"]) == 1

        result_artifact = completed_task["artifacts"][0]
        assert result_artifact["name"] == "final_result"
        assert result_artifact["parts"][0]["text"] == "Task completed successfully"


class TestAgentExecution:
    """Test agent execution functions"""

    def test_create_run_config_for_a2a_agent(self):
        """Test creating run configuration for A2A agent"""
        agent = create_a2a_agent("TestAgent", "Test agent", "You are helpful", [])

        model_provider = MockModelProvider()

        config = create_run_config_for_a2a_agent(agent, model_provider)

        assert isinstance(config, RunConfig)
        assert "TestAgent" in config.agent_registry
        assert config.model_provider == model_provider
        assert config.max_turns == 10
        assert config.on_event is not None

    def test_transform_to_run_state(self):
        """Test transforming agent state to run state"""
        message = create_user_message("Hello")
        agent_state = AgentState(
            sessionId="session_123",
            messages=[message],
            context={},
            artifacts=[],
            timestamp=datetime.now().isoformat(),
        )

        run_state = transform_to_run_state(agent_state, "TestAgent", {"user_id": "user_123"})

        assert run_state.current_agent_name == "TestAgent"
        assert len(run_state.messages) == 1
        assert run_state.messages[0] == message
        assert run_state.context["user_id"] == "user_123"
        assert run_state.turn_count == 0

    @pytest.mark.asyncio
    async def test_process_agent_query(self):
        """Test processing agent query"""
        agent = create_a2a_agent("TestAgent", "Test agent", "You are helpful", [])

        initial_state = create_initial_agent_state("session_123")
        model_provider = MockModelProvider()

        # Mock the run function to return a successful outcome
        class MockOutcome:
            def __init__(self):
                self.status = "completed"
                self.output = "Query processed successfully"

        class MockResult:
            def __init__(self):
                self.outcome = MockOutcome()

        with pytest.MonkeyPatch().context() as m:
            # Mock the run function
            async def mock_run(state, config):
                return MockResult()

            m.setattr("jaf.a2a.agent.run", mock_run)

            events = []
            async for event in process_agent_query(agent, "Hello", initial_state, model_provider):
                events.append(event)

            assert len(events) >= 1
            # Check that we have a final completion event
            final_event = events[-1]
            assert final_event.isTaskComplete is True
            assert "Query processed successfully" in final_event.content

    @pytest.mark.asyncio
    async def test_execute_a2a_agent_success(self):
        """Test successful A2A agent execution"""
        agent = create_a2a_agent("TestAgent", "Test agent", "You are helpful", [])

        context = {
            "message": {"parts": [{"kind": "text", "text": "Hello"}]},
            "session_id": "session_123",
        }

        model_provider = MockModelProvider()

        # Mock process_agent_query to return successful events
        async def mock_process_query(agent, query, state, provider):
            yield StreamEvent(
                isTaskComplete=True,
                content="Hello back!",
                new_state={},
                timestamp=datetime.now().isoformat(),
            )

        with pytest.MonkeyPatch().context() as m:
            m.setattr("jaf.a2a.agent.process_agent_query", mock_process_query)

            result = await execute_a2a_agent(context, agent, model_provider)

            assert "final_task" in result
            assert result["final_task"]["status"]["state"] == "completed"
            # Should have at least 1 artifact (may have more due to implementation details)
            assert len(result["final_task"]["artifacts"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_a2a_agent_error(self):
        """Test A2A agent execution with error"""
        agent = create_a2a_agent("TestAgent", "Test agent", "You are helpful", [])

        context = {
            "message": {"parts": [{"kind": "text", "text": "Hello"}]},
            "session_id": "session_123",
        }

        model_provider = MockModelProvider()

        # Mock process_agent_query to return error
        async def mock_process_query_error(agent, query, state, provider):
            yield StreamEvent(
                isTaskComplete=True,
                content="Error: Something went wrong",
                new_state={},
                timestamp=datetime.now().isoformat(),
            )

        with pytest.MonkeyPatch().context() as m:
            m.setattr("jaf.a2a.agent.process_agent_query", mock_process_query_error)

            result = await execute_a2a_agent(context, agent, model_provider)

            assert "error" in result
            assert result["error"] == "Something went wrong"
            assert result["final_task"]["status"]["state"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_a2a_agent_with_streaming(self):
        """Test A2A agent execution with streaming"""
        agent = create_a2a_agent("TestAgent", "Test agent", "You are helpful", [])

        context = {
            "message": {"parts": [{"kind": "text", "text": "Hello"}]},
            "session_id": "session_123",
        }

        model_provider = MockModelProvider()

        # Mock process_agent_query to return streaming events
        async def mock_process_streaming(agent, query, state, provider):
            # Working event
            yield StreamEvent(
                isTaskComplete=False,
                content="Processing...",
                updates="Working on task",
                new_state={},
                timestamp=datetime.now().isoformat(),
            )

            # Completion event
            yield StreamEvent(
                isTaskComplete=True,
                content="Task completed",
                new_state={},
                timestamp=datetime.now().isoformat(),
            )

        with pytest.MonkeyPatch().context() as m:
            m.setattr("jaf.a2a.agent.process_agent_query", mock_process_streaming)

            events = []
            async for event in execute_a2a_agent_with_streaming(context, agent, model_provider):
                events.append(event)

            # Should get: task submitted, working status, artifact, completion
            assert len(events) >= 3

            # Check event types
            kinds = [event["kind"] for event in events]
            assert "status-update" in kinds
            assert "artifact-update" in kinds

            # Check final status
            final_events = [e for e in events if e.get("final") is True]
            assert len(final_events) >= 1
            assert final_events[-1]["status"]["state"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__])
