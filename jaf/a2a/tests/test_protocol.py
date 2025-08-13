"""
Tests for A2A protocol handlers

Tests JSON-RPC protocol validation, request routing, and response handling
for the A2A implementation.
"""


import pytest

from jaf.a2a.protocol import (
    create_jsonrpc_error_response_dict,
    create_jsonrpc_success_response_dict,
    create_protocol_handler_config,
    handle_get_authenticated_extended_card,
    handle_message_send,
    handle_message_stream,
    handle_tasks_cancel,
    handle_tasks_get,
    map_error_to_a2a_error,
    route_a2a_request,
    validate_jsonrpc_request,
    validate_send_message_request,
)
from jaf.a2a.types import (
    A2AAgent,
    A2AAgentTool,
    A2AErrorCodes,
)


# Mock agent and tools for testing
async def mock_tool_execute(args, context):
    return {"result": "tool executed successfully"}


def create_mock_agent():
    """Create a mock A2A agent for testing"""
    tool = A2AAgentTool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object"},
        execute=mock_tool_execute
    )

    return A2AAgent(
        name="TestAgent",
        description="A test agent",
        supportedContentTypes=["text/plain"],
        instruction="You are a test agent",
        tools=[tool]
    )


class TestJSONRPCValidation:
    """Test JSON-RPC request validation"""

    def test_validate_jsonrpc_request_valid(self):
        """Test validation of valid JSON-RPC requests"""
        valid_request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {"test": "data"}
        }

        assert validate_jsonrpc_request(valid_request) is True

    def test_validate_jsonrpc_request_invalid(self):
        """Test validation of invalid JSON-RPC requests"""
        # Missing jsonrpc field
        invalid1 = {
            "id": "req_123",
            "method": "test",
            "params": {}
        }
        assert validate_jsonrpc_request(invalid1) is False

        # Wrong jsonrpc version
        invalid2 = {
            "jsonrpc": "1.0",
            "id": "req_123",
            "method": "test",
            "params": {}
        }
        assert validate_jsonrpc_request(invalid2) is False

        # Missing id
        invalid3 = {
            "jsonrpc": "2.0",
            "method": "test",
            "params": {}
        }
        assert validate_jsonrpc_request(invalid3) is False

        # Missing method
        invalid4 = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "params": {}
        }
        assert validate_jsonrpc_request(invalid4) is False

        # Non-dict input
        assert validate_jsonrpc_request("not a dict") is False
        assert validate_jsonrpc_request(None) is False

    def test_validate_send_message_request_valid(self):
        """Test validation of valid send message requests"""
        valid_request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        result = validate_send_message_request(valid_request)
        assert result["is_valid"] is True
        assert result["data"] == valid_request

    def test_validate_send_message_request_invalid(self):
        """Test validation of invalid send message requests"""
        # Invalid JSON-RPC base
        invalid1 = {
            "id": "req_123",
            "method": "message/send",
            "params": {"message": {}}
        }

        result1 = validate_send_message_request(invalid1)
        assert result1["is_valid"] is False
        assert result1["error"]["code"] == A2AErrorCodes.INVALID_REQUEST.value

        # Wrong method
        invalid2 = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "wrong/method",
            "params": {"message": {}}
        }

        result2 = validate_send_message_request(invalid2)
        assert result2["is_valid"] is False
        assert result2["error"]["code"] == A2AErrorCodes.METHOD_NOT_FOUND.value

        # Invalid params format
        invalid3 = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": "not a dict"
        }

        result3 = validate_send_message_request(invalid3)
        assert result3["is_valid"] is False
        assert result3["error"]["code"] == A2AErrorCodes.INVALID_PARAMS.value

        # Invalid message format
        invalid4 = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": "not a dict"
            }
        }

        result4 = validate_send_message_request(invalid4)
        assert result4["is_valid"] is False
        assert result4["error"]["code"] == A2AErrorCodes.INVALID_PARAMS.value

        # Wrong message kind
        invalid5 = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "wrong_kind",
                    "parts": []
                }
            }
        }

        result5 = validate_send_message_request(invalid5)
        assert result5["is_valid"] is False
        assert "Missing required message fields" in result5["error"]["message"]

        # Invalid parts format
        invalid6 = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "parts": "not a list"
                }
            }
        }

        result6 = validate_send_message_request(invalid6)
        assert result6["is_valid"] is False
        assert "Missing required message fields" in result6["error"]["message"]


class TestMessageHandlers:
    """Test message handling functions"""

    @pytest.mark.asyncio
    async def test_handle_message_send_success(self):
        """Test successful message send handling"""
        agent = create_mock_agent()

        # Mock executor function
        async def mock_executor(context, agent, model_provider):
            return {
                "final_task": {
                    "id": "task_123",
                    "status": {"state": "completed"},
                    "result": "Success"
                }
            }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        result = await handle_message_send(
            request, agent, None, mock_executor
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "result" in result
        assert result["result"]["id"] == "task_123"

    @pytest.mark.asyncio
    async def test_handle_message_send_error(self):
        """Test message send handling with executor error"""
        agent = create_mock_agent()

        # Mock executor function that returns error
        async def mock_executor_error(context, agent, model_provider):
            return {
                "error": "Something went wrong"
            }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        result = await handle_message_send(
            request, agent, None, mock_executor_error
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert result["error"]["message"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_handle_message_send_exception(self):
        """Test message send handling with executor exception"""
        agent = create_mock_agent()

        # Mock executor function that raises exception
        async def mock_executor_exception(context, agent, model_provider):
            raise ValueError("Test exception")

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        result = await handle_message_send(
            request, agent, None, mock_executor_exception
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert "Test exception" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_message_stream_success(self):
        """Test successful message stream handling"""
        agent = create_mock_agent()

        # Mock streaming executor function
        async def mock_streaming_executor(context, agent, model_provider):
            yield {"kind": "status-update", "status": "working"}
            yield {"kind": "status-update", "status": "completed"}

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        events = []
        async for event in handle_message_stream(
            request, agent, None, mock_streaming_executor
        ):
            events.append(event)

        assert len(events) == 2
        assert all(event["jsonrpc"] == "2.0" for event in events)
        assert all(event["id"] == "req_123" for event in events)
        assert events[0]["result"]["status"] == "working"
        assert events[1]["result"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_message_stream_exception(self):
        """Test message stream handling with exception"""
        agent = create_mock_agent()

        # Mock streaming executor that raises exception
        async def mock_streaming_executor_error(context, agent, model_provider):
            yield {"kind": "status-update", "status": "working"}
            raise RuntimeError("Stream error")

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        events = []
        async for event in handle_message_stream(
            request, agent, None, mock_streaming_executor_error
        ):
            events.append(event)

        # Should get one successful event and one error event
        assert len(events) == 2
        assert events[0]["result"]["status"] == "working"
        assert "error" in events[1]
        assert "Stream error" in events[1]["error"]["message"]


class TestTaskHandlers:
    """Test task management handlers"""

    @pytest.mark.asyncio
    async def test_handle_tasks_get_success(self):
        """Test successful task retrieval"""
        task_storage = {
            "task_123": {
                "id": "task_123",
                "status": {"state": "completed"},
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "agent", "content": "Hi there"}
                ],
                "artifacts": []
            }
        }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "tasks/get",
            "params": {
                "id": "task_123"
            }
        }

        result = await handle_tasks_get(request, task_storage)

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "result" in result
        assert result["result"]["id"] == "task_123"
        assert result["result"]["status"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_tasks_get_not_found(self):
        """Test task retrieval with non-existent task"""
        task_storage = {}

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "tasks/get",
            "params": {
                "id": "nonexistent_task"
            }
        }

        result = await handle_tasks_get(request, task_storage)

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert result["error"]["code"] == A2AErrorCodes.TASK_NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_handle_tasks_get_missing_id(self):
        """Test task retrieval with missing task ID"""
        task_storage = {}

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "tasks/get",
            "params": {}
        }

        result = await handle_tasks_get(request, task_storage)

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert result["error"]["code"] == A2AErrorCodes.INVALID_PARAMS.value
        assert "Task ID is required" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_handle_tasks_get_with_history_limit(self):
        """Test task retrieval with history length limit"""
        task_storage = {
            "task_123": {
                "id": "task_123",
                "status": {"state": "completed"},
                "history": [
                    {"role": "user", "content": "Message 1"},
                    {"role": "agent", "content": "Response 1"},
                    {"role": "user", "content": "Message 2"},
                    {"role": "agent", "content": "Response 2"},
                    {"role": "user", "content": "Message 3"},
                    {"role": "agent", "content": "Response 3"}
                ]
            }
        }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "tasks/get",
            "params": {
                "id": "task_123",
                "historyLength": 2
            }
        }

        result = await handle_tasks_get(request, task_storage)

        assert result["jsonrpc"] == "2.0"
        assert "result" in result
        # Should only have last 2 messages
        assert len(result["result"]["history"]) == 2
        assert result["result"]["history"][0]["content"] == "Message 3"
        assert result["result"]["history"][1]["content"] == "Response 3"

    @pytest.mark.asyncio
    async def test_handle_tasks_cancel_success(self):
        """Test successful task cancellation"""
        task_storage = {
            "task_123": {
                "id": "task_123",
                "status": {"state": "working"},
                "history": []
            }
        }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "tasks/cancel",
            "params": {
                "id": "task_123"
            }
        }

        result = await handle_tasks_cancel(request, task_storage)

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "result" in result
        assert result["result"]["status"]["state"] == "canceled"

    @pytest.mark.asyncio
    async def test_handle_tasks_cancel_not_cancelable(self):
        """Test task cancellation for non-cancelable task"""
        task_storage = {
            "task_123": {
                "id": "task_123",
                "status": {"state": "completed"},
                "history": []
            }
        }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "tasks/cancel",
            "params": {
                "id": "task_123"
            }
        }

        result = await handle_tasks_cancel(request, task_storage)

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert result["error"]["code"] == A2AErrorCodes.TASK_NOT_CANCELABLE.value


class TestAgentCardHandler:
    """Test agent card handler"""

    @pytest.mark.asyncio
    async def test_handle_get_authenticated_extended_card(self):
        """Test getting authenticated extended agent card"""
        agent_card = {
            "protocolVersion": "0.3.0",
            "name": "Test Agent",
            "description": "A test agent",
            "url": "http://localhost:3000/a2a",
            "skills": []
        }

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "agent/getAuthenticatedExtendedCard"
        }

        result = await handle_get_authenticated_extended_card(request, agent_card)

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "result" in result
        assert result["result"]["name"] == "Test Agent"
        assert result["result"]["protocolVersion"] == "0.3.0"


class TestRequestRouting:
    """Test request routing functionality"""

    @pytest.mark.asyncio
    async def test_route_a2a_request_message_send(self):
        """Test routing message/send requests"""
        agent = create_mock_agent()

        # Mock executor
        async def mock_executor(context, agent, model_provider):
            return {"final_task": {"id": "task_123", "status": "completed"}}

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg_123",
                    "contextId": "ctx_123",
                    "kind": "message"
                }
            }
        }

        result = await route_a2a_request(
            request, agent, None, {}, {}, mock_executor, None
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "result" in result

    @pytest.mark.asyncio
    async def test_route_a2a_request_invalid_method(self):
        """Test routing invalid method requests"""
        agent = create_mock_agent()

        request = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "method": "invalid/method",
            "params": {}
        }

        result = await route_a2a_request(
            request, agent, None, {}, {}, None, None
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert result["error"]["code"] == A2AErrorCodes.METHOD_NOT_FOUND.value

    @pytest.mark.asyncio
    async def test_route_a2a_request_invalid_jsonrpc(self):
        """Test routing invalid JSON-RPC requests"""
        agent = create_mock_agent()

        invalid_request = {
            "id": "req_123",
            "method": "message/send"
            # Missing jsonrpc field
        }

        result = await route_a2a_request(
            invalid_request, agent, None, {}, {}, None, None
        )

        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req_123"
        assert "error" in result
        assert result["error"]["code"] == A2AErrorCodes.INVALID_REQUEST.value


class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_jsonrpc_success_response_dict(self):
        """Test creating JSON-RPC success response dict"""
        response = create_jsonrpc_success_response_dict(
            "req_123",
            {"status": "success"}
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "req_123"
        assert response["result"] == {"status": "success"}

    def test_create_jsonrpc_error_response_dict(self):
        """Test creating JSON-RPC error response dict"""
        error = {
            "code": -32603,
            "message": "Internal error"
        }

        response = create_jsonrpc_error_response_dict("req_123", error)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "req_123"
        assert response["error"] == error

    def test_map_error_to_a2a_error(self):
        """Test mapping Python exceptions to A2A errors"""
        # Test with ValueError
        value_error = ValueError("Invalid value")
        error_dict = map_error_to_a2a_error(value_error)

        assert error_dict["code"] == A2AErrorCodes.INTERNAL_ERROR.value
        assert error_dict["message"] == "Invalid value"
        assert error_dict["data"]["type"] == "ValueError"

        # Test with generic exception
        runtime_error = RuntimeError("Something went wrong")
        error_dict2 = map_error_to_a2a_error(runtime_error)

        assert error_dict2["code"] == A2AErrorCodes.INTERNAL_ERROR.value
        assert error_dict2["message"] == "Something went wrong"
        assert error_dict2["data"]["type"] == "RuntimeError"

    def test_create_protocol_handler_config(self):
        """Test creating protocol handler configuration"""
        agents = {"TestAgent": create_mock_agent()}
        agent_card = {"name": "Test Server"}

        async def mock_executor(context, agent, model_provider):
            return {"result": "success"}

        async def mock_streaming_executor(context, agent, model_provider):
            yield {"event": "stream"}

        config = create_protocol_handler_config(
            agents, None, agent_card, mock_executor, mock_streaming_executor
        )

        assert "agents" in config
        assert "model_provider" in config
        assert "agent_card" in config
        assert "task_storage" in config
        assert "handle_request" in config

        assert config["agents"] == agents
        assert config["agent_card"] == agent_card
        assert isinstance(config["task_storage"], dict)


if __name__ == "__main__":
    pytest.main([__file__])
