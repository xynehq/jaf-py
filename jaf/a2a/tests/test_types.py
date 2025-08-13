"""
Tests for A2A core types and models

Tests all Pydantic models, factory functions, and type validation
for the A2A protocol implementation.
"""

import json
from datetime import datetime

import pytest

from jaf.a2a.types import (
    A2AAgent,
    A2AClientConfig,
    A2AClientState,
    A2AErrorCodes,
    # Core A2A types
    A2AMessage,
    A2ATaskStatus,
    A2AToolResult,
    AgentCapabilities,
    # Agent and configuration types
    AgentCard,
    AgentProvider,
    AgentSkill,
    # Stream and state types
    AgentState,
    MessageSendConfiguration,
    SendMessageRequest,
    StreamEvent,
    ToolContext,
    create_a2a_agent_tool,
    create_a2a_artifact,
    create_a2a_data_part,
    create_a2a_error,
    # Factory functions
    create_a2a_message,
    create_a2a_task,
    create_a2a_text_part,
    create_jsonrpc_error_response,
    create_jsonrpc_request,
    create_jsonrpc_success_response,
)


class TestA2AParts:
    """Test A2A part types"""

    def test_create_text_part(self):
        """Test text part creation"""
        part = create_a2a_text_part("Hello world")

        assert part.kind == "text"
        assert part.text == "Hello world"
        assert part.model_dump() == {
            "kind": "text",
            "text": "Hello world"
        }

    def test_create_data_part(self):
        """Test data part creation"""
        data = {"key": "value", "number": 42}
        part = create_a2a_data_part(data)

        assert part.kind == "data"
        assert part.data == data
        assert part.model_dump() == {
            "kind": "data",
            "data": data
        }

    def test_part_immutability(self):
        """Test that parts are immutable"""
        part = create_a2a_text_part("test")

        with pytest.raises(Exception):  # ValidationError for frozen model
            part.text = "modified"


class TestA2AMessage:
    """Test A2A message types"""

    def test_create_a2a_message(self):
        """Test A2A message creation"""
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Hello")],
            context_id="ctx_123"
        )

        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].text == "Hello"
        assert message.context_id == "ctx_123"
        assert message.kind == "message"
        assert message.message_id.startswith("msg_")

    def test_message_validation(self):
        """Test message validation"""
        # Valid message
        message = A2AMessage(
            role="user",
            parts=[create_a2a_text_part("test")],
            messageId="msg_123",
            contextId="ctx_123",
            kind="message"
        )
        assert message.role == "user"

        # Invalid role
        with pytest.raises(Exception):
            A2AMessage(
                role="invalid",
                parts=[create_a2a_text_part("test")],
                messageId="msg_123",
                contextId="ctx_123",
                kind="message"
            )

    def test_message_serialization(self):
        """Test message JSON serialization"""
        message = create_a2a_message(
            role="agent",
            parts=[
                create_a2a_text_part("Response"),
                create_a2a_data_part({"result": "success"})
            ],
            context_id="ctx_456"
        )

        # Should serialize to dict
        data = message.model_dump()
        assert isinstance(data, dict)
        assert data["role"] == "agent"
        assert len(data["parts"]) == 2

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Should deserialize back
        restored_data = json.loads(json_str)
        # The message_id field should be present in the serialized data (snake_case)
        assert "message_id" in restored_data
        # Convert to camelCase for deserialization
        if "message_id" in restored_data:
            restored_data["messageId"] = restored_data.pop("message_id")
        if "context_id" in restored_data:
            restored_data["contextId"] = restored_data.pop("context_id")
        if "task_id" in restored_data:
            restored_data["taskId"] = restored_data.pop("task_id")
        if "reference_task_ids" in restored_data:
            restored_data["referenceTaskIds"] = restored_data.pop("reference_task_ids")
        restored_message = A2AMessage(**restored_data)
        assert restored_message.role == message.role


class TestA2ATask:
    """Test A2A task types"""

    def test_create_a2a_task(self):
        """Test A2A task creation"""
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Task request")],
            context_id="ctx_123"
        )

        task = create_a2a_task(
            initial_message=message,
            context_id="ctx_123"
        )

        assert task.context_id == "ctx_123"
        assert task.status.state == "submitted"
        assert len(task.history) == 1
        assert task.history[0] == message
        assert task.kind == "task"
        assert task.id.startswith("task_")

    def test_task_status_updates(self):
        """Test task status progression"""
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Test")],
            context_id="ctx_123"
        )

        task = create_a2a_task(message, "ctx_123")

        # Initial status
        assert task.status.state == "submitted"
        assert task.status.timestamp is not None

        # Update to working
        working_status = A2ATaskStatus(
            state="working",
            timestamp=datetime.now().isoformat()
        )

        # Task should be immutable, so we create new one
        working_task = task.model_copy(update={"status": working_status})
        assert working_task.status.state == "working"
        assert task.status.state == "submitted"  # Original unchanged

    def test_task_with_artifacts(self):
        """Test task with artifacts"""
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Create artifact")],
            context_id="ctx_123"
        )

        artifact = create_a2a_artifact(
            name="test_artifact",
            parts=[create_a2a_text_part("Artifact content")]
        )

        task = create_a2a_task(message, "ctx_123")
        task_with_artifact = task.model_copy(
            update={"artifacts": [artifact]}
        )

        assert len(task_with_artifact.artifacts) == 1
        assert task_with_artifact.artifacts[0].name == "test_artifact"


class TestJSONRPCTypes:
    """Test JSON-RPC protocol types"""

    def test_create_jsonrpc_request(self):
        """Test JSON-RPC request creation"""
        request = create_jsonrpc_request(
            method="message/send",
            params={"test": "data"},
            request_id="req_123"
        )

        assert request.jsonrpc == "2.0"
        assert request.method == "message/send"
        assert request.params == {"test": "data"}
        assert request.id == "req_123"

    def test_create_jsonrpc_success_response(self):
        """Test JSON-RPC success response creation"""
        response = create_jsonrpc_success_response(
            id="req_123",
            result={"success": True}
        )

        assert response.jsonrpc == "2.0"
        assert response.id == "req_123"
        assert response.result == {"success": True}

    def test_create_jsonrpc_error_response(self):
        """Test JSON-RPC error response creation"""
        error = create_a2a_error(
            code=A2AErrorCodes.INVALID_REQUEST,
            message="Invalid request format"
        )

        response = create_jsonrpc_error_response(
            id="req_123",
            error=error
        )

        assert response.jsonrpc == "2.0"
        assert response.id == "req_123"
        assert response.error.code == A2AErrorCodes.INVALID_REQUEST.value
        assert response.error.message == "Invalid request format"

    def test_message_send_request(self):
        """Test SendMessageRequest creation"""
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Hello")],
            context_id="ctx_123"
        )

        request = SendMessageRequest(
            jsonrpc="2.0",
            id="req_123",
            method="message/send",
            params={
                "message": message,
                "configuration": MessageSendConfiguration(
                    model="gpt-4",
                    temperature=0.7
                )
            }
        )

        assert request.params.message.role == "user"
        assert request.params.configuration.model == "gpt-4"
        assert request.params.configuration.temperature == 0.7


class TestAgentTypes:
    """Test agent-related types"""

    def test_create_a2a_agent_tool(self):
        """Test A2A agent tool creation"""
        async def mock_execute(args, context):
            return {"result": "success"}

        tool = create_a2a_agent_tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            execute_func=mock_execute
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.execute == mock_execute

    def test_a2a_agent_creation(self):
        """Test A2A agent creation"""
        async def mock_execute(args, context):
            return {"result": "tool executed"}

        tool = create_a2a_agent_tool(
            name="helper_tool",
            description="Helper tool",
            parameters={},
            execute_func=mock_execute
        )

        agent = A2AAgent(
            name="TestAgent",
            description="A test agent",
            supportedContentTypes=["text/plain"],
            instruction="You are a test agent",
            tools=[tool]
        )

        assert agent.name == "TestAgent"
        assert agent.description == "A test agent"
        assert "text/plain" in agent.supported_content_types
        assert agent.instruction == "You are a test agent"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "helper_tool"

    def test_agent_card_structure(self):
        """Test agent card structure"""
        skills = [
            AgentSkill(
                id="skill_1",
                name="Test Skill",
                description="A test skill",
                tags=["test"],
                examples=["Example usage"],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            )
        ]

        capabilities = AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=True
        )

        provider = AgentProvider(
            organization="Test Org",
            url="https://test.com"
        )

        card = AgentCard(
            protocolVersion="0.3.0",
            name="Test Agent",
            description="Test agent description",
            url="http://localhost:3000/a2a",
            preferredTransport="JSONRPC",
            version="1.0.0",
            provider=provider,
            capabilities=capabilities,
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            skills=skills
        )

        assert card.protocol_version == "0.3.0"
        assert card.name == "Test Agent"
        assert card.capabilities.streaming is True
        assert len(card.skills) == 1
        assert card.skills[0].name == "Test Skill"


class TestStreamTypes:
    """Test streaming and event types"""

    def test_stream_event_creation(self):
        """Test stream event creation"""
        event = StreamEvent(
            isTaskComplete=False,
            content="Processing...",
            updates="Working on task",
            newState={"status": "working"},
            timestamp=datetime.now().isoformat()
        )

        assert event.isTaskComplete is False
        assert event.content == "Processing..."
        assert event.updates == "Working on task"
        assert event.new_state == {"status": "working"}

    def test_a2a_stream_event(self):
        """Test A2A stream event"""
        from jaf.a2a.types import A2AStatusUpdateEvent, A2ATaskStatus

        status = A2ATaskStatus(
            state="working",
            timestamp="2025-08-12T22:52:29.159776"
        )

        event = A2AStatusUpdateEvent(
            kind="status-update",
            taskId="task_123",
            contextId="ctx_456",
            status=status,
            final=False
        )

        assert event.kind == "status-update"
        assert event.task_id == "task_123"
        assert event.context_id == "ctx_456"
        assert event.final is False

    def test_agent_state(self):
        """Test agent state management"""
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Test")],
            context_id="ctx_123"
        )

        state = AgentState(
            sessionId="session_123",
            messages=[message],
            context={"key": "value"},
            artifacts=[],
            timestamp=datetime.now().isoformat()
        )

        assert state.sessionId == "session_123"
        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.context == {"key": "value"}


class TestClientTypes:
    """Test client configuration types"""

    def test_client_config(self):
        """Test A2A client configuration"""
        config = A2AClientConfig(
            baseUrl="http://localhost:3000",
            timeout=30000
        )

        assert config.base_url == "http://localhost:3000"
        assert config.timeout == 30000

    def test_client_state(self):
        """Test A2A client state"""
        config = A2AClientConfig(
            baseUrl="http://localhost:3000",
            timeout=30000
        )

        state = A2AClientState(
            config=config,
            sessionId="session_123"
        )

        assert state.config.base_url == "http://localhost:3000"
        assert state.session_id == "session_123"

    def test_tool_context(self):
        """Test tool context"""
        context = ToolContext(
            actions={
                "requiresInput": False,
                "skipSummarization": True,
                "escalate": False
            },
            metadata={"user_id": "user_123"}
        )

        assert context.actions["requiresInput"] is False
        assert context.actions["skipSummarization"] is True
        assert context.metadata["user_id"] == "user_123"


class TestErrorTypes:
    """Test error handling types"""

    def test_a2a_error_codes(self):
        """Test A2A error codes"""
        assert A2AErrorCodes.INVALID_REQUEST.value == -32600
        assert A2AErrorCodes.METHOD_NOT_FOUND.value == -32601
        assert A2AErrorCodes.INVALID_PARAMS.value == -32602
        assert A2AErrorCodes.INTERNAL_ERROR.value == -32603

    def test_create_a2a_error(self):
        """Test A2A error creation"""
        error = create_a2a_error(
            code=A2AErrorCodes.INVALID_PARAMS,
            message="Invalid parameters provided",
            data={"field": "missing_field"}
        )

        assert error.code == A2AErrorCodes.INVALID_PARAMS.value
        assert error.message == "Invalid parameters provided"
        assert error.data == {"field": "missing_field"}

    def test_a2a_tool_result(self):
        """Test A2A tool result types"""
        # Success result
        success_result = A2AToolResult(
            status="success",
            result={"result": "completed"},
            data={"result": "completed"},
            error=None
        )

        assert success_result.status == "success"
        assert success_result.data == {"result": "completed"}
        assert success_result.error is None

        # Error result
        error_result = A2AToolResult(
            status="error",
            result=None,
            data=None,
            error=create_a2a_error(
                code=A2AErrorCodes.INTERNAL_ERROR,
                message="Tool execution failed"
            )
        )

        assert error_result.status == "error"
        assert error_result.data is None
        assert error_result.error.message == "Tool execution failed"


class TestFactoryFunctions:
    """Test factory functions for creating types"""

    def test_all_factory_functions(self):
        """Test that all factory functions work correctly"""

        # Text and data parts
        text_part = create_a2a_text_part("Hello")
        data_part = create_a2a_data_part({"key": "value"})

        # Message
        message = create_a2a_message(
            role="user",
            parts=[text_part, data_part],
            context_id="ctx_123"
        )

        # Task
        task = create_a2a_task(message, "ctx_123")

        # Artifact
        artifact = create_a2a_artifact(
            name="test_artifact",
            parts=[text_part]
        )

        # JSON-RPC request
        request = create_jsonrpc_request(
            method="test/method",
            params={"test": True}
        )

        # JSON-RPC responses
        success_response = create_jsonrpc_success_response(
            id="req_123",
            result={"success": True}
        )

        error = create_a2a_error(
            code=A2AErrorCodes.INTERNAL_ERROR,
            message="Test error"
        )

        error_response = create_jsonrpc_error_response(
            id="req_123",
            error=error
        )

        # Verify all created correctly
        assert text_part.kind == "text"
        assert data_part.kind == "data"
        assert message.role == "user"
        assert task.kind == "task"
        assert artifact.name == "test_artifact"
        assert request.method == "test/method"
        assert success_response.result == {"success": True}
        assert error_response.error.message == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])
