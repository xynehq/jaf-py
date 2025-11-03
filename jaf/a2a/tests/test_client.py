"""
Tests for A2A client functionality

Tests the httpx-based client library for A2A communication,
including request creation, response handling, and streaming.
"""

from unittest.mock import AsyncMock, patch

import pytest

from jaf.a2a.client import (
    check_a2a_health,
    connect_to_a2a_agent,
    create_a2a_client,
    create_a2a_message_dict,
    create_message_request,
    create_streaming_message_request,
    discover_agents,
    extract_text_response,
    get_a2a_capabilities,
    get_agent_card,
    parse_sse_event,
    send_a2a_request,
    send_http_request,
    send_message,
    send_message_to_agent,
    validate_a2a_response,
)
from jaf.a2a.types import A2AClientState


class TestClientCreation:
    """Test A2A client creation and configuration"""

    def test_create_a2a_client_basic(self):
        """Test basic client creation"""
        client = create_a2a_client("http://localhost:3000")

        assert isinstance(client, A2AClientState)
        assert client.config.base_url == "http://localhost:3000"
        assert client.config.timeout == 30000  # Default timeout
        assert client.session_id.startswith("client_")

    def test_create_a2a_client_with_config(self):
        """Test client creation with custom configuration"""
        config = {"timeout": 60000, "custom_field": "value"}

        client = create_a2a_client("http://example.com/", config)

        assert client.config.base_url == "http://example.com"  # Trailing slash removed
        assert client.config.timeout == 60000
        assert client.session_id.startswith("client_")

    def test_create_a2a_client_trailing_slash(self):
        """Test that trailing slashes are removed from base URL"""
        client1 = create_a2a_client("http://localhost:3000/")
        client2 = create_a2a_client("http://localhost:3000")

        assert client1.config.base_url == "http://localhost:3000"
        assert client2.config.base_url == "http://localhost:3000"


class TestRequestCreation:
    """Test request creation functions"""

    def test_create_message_request(self):
        """Test creating message requests"""
        request = create_message_request("Hello, world!", "session_123", {"model": "gpt-4"})

        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "message/send"
        assert "id" in request
        assert request["id"].startswith("req_")

        params = request["params"]
        assert params["configuration"] == {"model": "gpt-4"}

        message = params["message"]
        assert message["role"] == "user"
        assert message["contextId"] == "session_123"
        assert message["kind"] == "message"
        assert len(message["parts"]) == 1
        assert message["parts"][0]["kind"] == "text"
        assert message["parts"][0]["text"] == "Hello, world!"

    def test_create_message_request_no_config(self):
        """Test creating message request without configuration"""
        request = create_message_request("Test message", "session_456")

        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "message/send"
        assert request["params"]["configuration"] is None
        assert request["params"]["message"]["parts"][0]["text"] == "Test message"

    def test_create_streaming_message_request(self):
        """Test creating streaming message requests"""
        request = create_streaming_message_request(
            "Stream this message", "session_789", {"temperature": 0.8}
        )

        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "message/stream"
        assert "id" in request

        params = request["params"]
        assert params["configuration"] == {"temperature": 0.8}

        message = params["message"]
        assert message["role"] == "user"
        assert message["contextId"] == "session_789"
        assert message["parts"][0]["text"] == "Stream this message"


class TestHTTPRequests:
    """Test HTTP request functions"""

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.httpx.AsyncClient")
    async def test_send_http_request_success(self, mock_client_class):
        """Test successful HTTP request"""
        # Mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"result": "success"}

        # Mock client
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        url = "http://localhost:3000/a2a"
        body = {"test": "data"}

        result = await send_http_request(url, body, 30000)

        # Await the mock response if it's a coroutine
        expected_result = {"result": "success"}
        if hasattr(result, "__await__"):
            result = await result
        assert result == expected_result
        mock_client.post.assert_called_once_with(
            url,
            json=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.httpx.AsyncClient")
    async def test_send_http_request_http_error(self, mock_client_class):
        """Test HTTP request with error status"""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.is_success = False
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        # Mock client
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            await send_http_request("http://localhost:3000/a2a", {}, 30000)

        assert "HTTP 404: Not Found" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.httpx.AsyncClient")
    async def test_send_http_request_timeout(self, mock_client_class):
        """Test HTTP request timeout"""
        # Mock client that raises timeout
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("TimeoutException")  # httpx.TimeoutException
        mock_client_class.return_value.__aenter__.return_value = mock_client

        with pytest.raises(Exception) as exc_info:
            await send_http_request("http://localhost:3000/a2a", {}, 5000)

        assert "TimeoutException" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_a2a_request(self):
        """Test A2A request wrapper"""
        client = create_a2a_client("http://localhost:3000")
        request = {"test": "request"}

        with patch("jaf.a2a.client.send_http_request") as mock_send:
            mock_send.return_value = {"result": "success"}

            result = await send_a2a_request(client, request)

            assert result == {"result": "success"}
            mock_send.assert_called_once_with("http://localhost:3000/a2a", request, 30000)


class TestMessageSending:
    """Test message sending functions"""

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful message sending"""
        client = create_a2a_client("http://localhost:3000")

        # Mock successful response
        mock_response = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "result": {
                "kind": "task",
                "artifacts": [{"parts": [{"kind": "text", "text": "Hello back!"}]}],
            },
        }

        with patch("jaf.a2a.client.send_a2a_request") as mock_send:
            mock_send.return_value = mock_response

            result = await send_message(client, "Hello!", {"model": "gpt-4"})

            assert result == "Hello back!"
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_error(self):
        """Test message sending with error response"""
        client = create_a2a_client("http://localhost:3000")

        # Mock error response
        mock_response = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "error": {"code": -32603, "message": "Internal error"},
        }

        with patch("jaf.a2a.client.send_a2a_request") as mock_send:
            mock_send.return_value = mock_response

            with pytest.raises(Exception) as exc_info:
                await send_message(client, "Hello!")

            assert "A2A Error -32603: Internal error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_message_to_agent(self):
        """Test sending message to specific agent"""
        client = create_a2a_client("http://localhost:3000")

        mock_response = {"jsonrpc": "2.0", "id": "req_123", "result": "Agent response"}

        with patch("jaf.a2a.client.send_http_request") as mock_send:
            mock_send.return_value = mock_response

            result = await send_message_to_agent(client, "TestAgent", "Hello agent!")

            assert result == "Agent response"
            mock_send.assert_called_once()

            # Check that correct URL was used
            call_args = mock_send.call_args[0]
            assert call_args[0] == "http://localhost:3000/a2a/agents/TestAgent"


class TestStreaming:
    """Test streaming functionality"""

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.stream_message")
    async def test_stream_message_success(self, mock_stream_message):
        """Test successful message streaming"""
        client = create_a2a_client("http://localhost:3000")

        # Mock the stream_message function to return test events
        async def mock_stream_events(client, message, config=None):
            yield {"status": "working"}
            yield {"status": "completed"}

        mock_stream_message.side_effect = mock_stream_events

        events = []
        async for event in mock_stream_message(client, "Stream test"):
            events.append(event)

        assert len(events) == 2
        assert events[0]["status"] == "working"
        assert events[1]["status"] == "completed"
        mock_stream_message.assert_called_once_with(client, "Stream test")

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.stream_message_to_agent")
    async def test_stream_message_to_agent(self, mock_stream_message_to_agent):
        """Test streaming message to specific agent"""
        client = create_a2a_client("http://localhost:3000")

        # Mock the stream_message_to_agent function to return test events
        async def mock_stream_events(client, agent_name, message, config=None):
            yield {"event": "test"}

        mock_stream_message_to_agent.side_effect = mock_stream_events

        events = []
        async for event in mock_stream_message_to_agent(client, "TestAgent", "Stream to agent"):
            events.append(event)

        assert len(events) == 1
        assert events[0]["event"] == "test"
        mock_stream_message_to_agent.assert_called_once_with(client, "TestAgent", "Stream to agent")


class TestAgentDiscovery:
    """Test agent discovery functions"""

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.httpx.AsyncClient")
    async def test_get_agent_card(self, mock_client_class):
        """Test getting agent card"""
        client = create_a2a_client("http://localhost:3000")

        mock_agent_card = {
            "protocolVersion": "0.3.0",
            "name": "Test Agent",
            "description": "A test agent",
            "skills": [],
        }

        # Mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json.return_value = mock_agent_card

        # Mock client
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await get_agent_card(client)

        # Handle mock coroutine if needed
        if hasattr(result, "__await__"):
            result = await result
        assert result == mock_agent_card
        mock_client.get.assert_called_once_with(
            "http://localhost:3000/.well-known/agent-card", headers={"Accept": "application/json"}
        )

    @pytest.mark.asyncio
    async def test_discover_agents(self):
        """Test discover agents convenience function"""
        mock_agent_card = {"name": "Test Server", "skills": [{"name": "skill1"}]}

        with patch("jaf.a2a.client.get_agent_card") as mock_get_card:
            mock_get_card.return_value = mock_agent_card

            result = await discover_agents("http://localhost:3000")

            assert result == mock_agent_card
            mock_get_card.assert_called_once()


class TestHealthAndCapabilities:
    """Test health and capabilities functions"""

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.httpx.AsyncClient")
    async def test_check_a2a_health(self, mock_client_class):
        """Test health check"""
        client = create_a2a_client("http://localhost:3000")

        mock_health = {"status": "healthy", "protocol": "A2A", "version": "0.3.0"}

        # Mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json.return_value = mock_health

        # Mock client
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await check_a2a_health(client)

        # Handle mock coroutine if needed
        if hasattr(result, "__await__"):
            result = await result
        assert result == mock_health
        mock_client.get.assert_called_once_with(
            "http://localhost:3000/a2a/health", headers={"Accept": "application/json"}
        )

    @pytest.mark.asyncio
    @patch("jaf.a2a.client.httpx.AsyncClient")
    async def test_get_a2a_capabilities(self, mock_client_class):
        """Test getting capabilities"""
        client = create_a2a_client("http://localhost:3000")

        mock_capabilities = {
            "supportedMethods": ["message/send", "message/stream"],
            "supportedTransports": ["JSONRPC"],
        }

        # Mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json.return_value = mock_capabilities

        # Mock client
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await get_a2a_capabilities(client)

        # Handle mock coroutine if needed
        if hasattr(result, "__await__"):
            result = await result
        assert result == mock_capabilities
        mock_client.get.assert_called_once_with(
            "http://localhost:3000/a2a/capabilities", headers={"Accept": "application/json"}
        )


class TestResponseProcessing:
    """Test response processing functions"""

    def test_extract_text_response_string(self):
        """Test extracting text from string response"""
        result = extract_text_response("Simple string response")
        assert result == "Simple string response"

    def test_extract_text_response_task(self):
        """Test extracting text from task response"""
        task_response = {
            "kind": "task",
            "artifacts": [
                {
                    "parts": [
                        {"kind": "text", "text": "First text part"},
                        {"kind": "data", "data": {"key": "value"}},
                        {"kind": "text", "text": "Second text part"},
                    ]
                }
            ],
            "history": [{"parts": [{"kind": "text", "text": "History message"}]}],
        }

        result = extract_text_response(task_response)
        assert result == "First text part"

    def test_extract_text_response_task_no_artifacts(self):
        """Test extracting text from task with no artifacts"""
        task_response = {
            "kind": "task",
            "artifacts": [],
            "history": [
                {
                    "parts": [
                        {"kind": "text", "text": "History text 1"},
                        {"kind": "text", "text": "History text 2"},
                    ]
                }
            ],
        }

        result = extract_text_response(task_response)
        assert result == "History text 1\nHistory text 2"

    def test_extract_text_response_message(self):
        """Test extracting text from message response"""
        message_response = {
            "kind": "message",
            "parts": [
                {"kind": "text", "text": "Message part 1"},
                {"kind": "text", "text": "Message part 2"},
            ],
        }

        result = extract_text_response(message_response)
        assert result == "Message part 1\nMessage part 2"

    def test_extract_text_response_dict(self):
        """Test extracting text from generic dict"""
        dict_response = {"key": "value", "number": 42}
        result = extract_text_response(dict_response)

        # Should return JSON representation
        assert '"key": "value"' in result
        assert '"number": 42' in result

    def test_extract_text_response_no_content(self):
        """Test extracting text when no content available"""
        result = extract_text_response(None)
        assert result == "No response content available"

        result2 = extract_text_response({})
        assert result2 == "{}"


class TestUtilityFunctions:
    """Test utility functions"""

    def test_create_a2a_message_dict(self):
        """Test creating A2A message dictionary"""
        message = create_a2a_message_dict("Hello world", "agent", "ctx_123")

        assert message["role"] == "agent"
        assert message["parts"][0]["kind"] == "text"
        assert message["parts"][0]["text"] == "Hello world"
        assert message["contextId"] == "ctx_123"
        assert message["kind"] == "message"
        assert message["messageId"].startswith("msg_")

    def test_create_a2a_message_dict_defaults(self):
        """Test creating A2A message with defaults"""
        message = create_a2a_message_dict("Test message")

        assert message["role"] == "user"
        assert message["contextId"] is None
        assert message["parts"][0]["text"] == "Test message"

    def test_parse_sse_event_valid(self):
        """Test parsing valid SSE events"""
        # Valid SSE event
        line = 'data: {"event": "test", "data": "value"}'
        result = parse_sse_event(line)

        assert result == {"event": "test", "data": "value"}

    def test_parse_sse_event_invalid(self):
        """Test parsing invalid SSE events"""
        # Not SSE format
        assert parse_sse_event("regular line") is None

        # Empty data
        assert parse_sse_event("data: ") is None
        assert parse_sse_event("data:    ") is None

        # Invalid JSON
        assert parse_sse_event("data: invalid json") is None

    def test_validate_a2a_response_valid(self):
        """Test validating valid A2A responses"""
        valid_response = {"jsonrpc": "2.0", "id": "req_123", "result": {"data": "success"}}

        assert validate_a2a_response(valid_response) is True

        # Valid error response
        valid_error = {
            "jsonrpc": "2.0",
            "id": "req_123",
            "error": {"code": -32603, "message": "Error"},
        }

        assert validate_a2a_response(valid_error) is True

    def test_validate_a2a_response_invalid(self):
        """Test validating invalid A2A responses"""
        # Missing jsonrpc
        assert validate_a2a_response({"id": "req_123", "result": {}}) is False

        # Wrong jsonrpc version
        assert validate_a2a_response({"jsonrpc": "1.0", "id": "req_123", "result": {}}) is False

        # Missing id
        assert validate_a2a_response({"jsonrpc": "2.0", "result": {}}) is False

        # Missing both result and error
        assert validate_a2a_response({"jsonrpc": "2.0", "id": "req_123"}) is False

        # Not a dict
        assert validate_a2a_response("not a dict") is False


class TestConvenienceConnection:
    """Test convenience connection function"""

    @pytest.mark.asyncio
    async def test_connect_to_a2a_agent(self):
        """Test convenience connection to A2A agent"""
        mock_agent_card = {"name": "Test Agent", "description": "Test description"}

        with patch("jaf.a2a.client.get_agent_card") as mock_get_card:
            mock_get_card.return_value = mock_agent_card

            connection = await connect_to_a2a_agent("http://localhost:3000")

            assert "client" in connection
            assert "agent_card" in connection
            assert "ask" in connection
            assert "stream" in connection
            assert "health" in connection
            assert "capabilities" in connection

            assert connection["agent_card"] == mock_agent_card
            assert isinstance(connection["client"], A2AClientState)

            # Test that the convenience methods are callable
            assert callable(connection["ask"])
            assert callable(connection["stream"])
            assert callable(connection["health"])
            assert callable(connection["capabilities"])


if __name__ == "__main__":
    pytest.main([__file__])
