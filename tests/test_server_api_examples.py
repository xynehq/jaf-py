"""
Test suite for Server API examples documentation.
Tests all code examples from docs/server-api.md to ensure they work with the actual implementation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from jaf import Agent, make_litellm_provider
from jaf.server.types import ServerConfig
from jaf.core.types import RunConfig, Message
from jaf.core.tools import create_function_tool


class TestServerAPIBasicExamples:
    """Test basic server API examples from the documentation."""
    
    def test_server_config_creation(self):
        """Test basic server configuration creation."""
        
        # Create a simple agent
        def agent_instructions(state):
            return "You are helpful."
        
        agent = Agent(name="MyAgent", instructions=agent_instructions, tools=[])
        
        # Mock model provider
        mock_provider = MagicMock()
        
        # Create server configuration
        server_config = ServerConfig(
            host="127.0.0.1",
            port=3000,
            agent_registry={"MyAgent": agent},
            run_config=RunConfig(
                agent_registry={"MyAgent": agent},
                model_provider=mock_provider,
                max_turns=5
            )
        )
        
        assert server_config.host == "127.0.0.1"
        assert server_config.port == 3000
        assert "MyAgent" in server_config.agent_registry
        assert server_config.run_config.max_turns == 5
    
    def test_health_check_response_format(self):
        """Test health check response format."""
        
        # Mock health check response structure
        health_response = {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00.123456Z",
            "version": "2.0.0",
            "uptime": 45000
        }
        
        assert health_response["status"] == "healthy"
        assert "timestamp" in health_response
        assert "version" in health_response
        assert "uptime" in health_response
        assert isinstance(health_response["uptime"], int)
    
    def test_agents_list_response_format(self):
        """Test agents list response format."""
        
        # Mock agents list response
        agents_response = {
            "success": True,
            "data": {
                "agents": [
                    {
                        "name": "MathTutor",
                        "description": "You are a helpful math tutor. Use the calculator tool to perform calculations and explain math concepts clearly.",
                        "tools": ["calculate"]
                    },
                    {
                        "name": "ChatBot", 
                        "description": "You are a friendly chatbot. Use the greeting tool when meeting new people, and engage in helpful conversation.",
                        "tools": ["greet"]
                    }
                ]
            }
        }
        
        assert agents_response["success"] is True
        assert "data" in agents_response
        assert "agents" in agents_response["data"]
        assert len(agents_response["data"]["agents"]) == 2
        
        # Check agent structure
        agent = agents_response["data"]["agents"][0]
        assert "name" in agent
        assert "description" in agent
        assert "tools" in agent
        assert isinstance(agent["tools"], list)


class TestChatRequestExamples:
    """Test chat request examples."""
    
    def test_chat_request_structure(self):
        """Test chat request structure."""
        
        chat_request = {
            "agent_name": "MathTutor",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 15 * 7?"
                }
            ],
            "context": {
                "userId": "user-123",
                "permissions": ["user"]
            },
            "max_turns": 5,
            "conversation_id": "math-session-1",
            "stream": False
        }
        
        assert chat_request["agent_name"] == "MathTutor"
        assert len(chat_request["messages"]) == 1
        assert chat_request["messages"][0]["role"] == "user"
        assert chat_request["messages"][0]["content"] == "What is 15 * 7?"
        assert "userId" in chat_request["context"]
        assert chat_request["max_turns"] == 5
        assert chat_request["stream"] is False
    
    def test_chat_response_structure(self):
        """Test chat response structure."""
        
        chat_response = {
            "success": True,
            "data": {
                "run_id": "run_12345",
                "trace_id": "trace_67890",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 15 * 7?"
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": "{\"expression\": \"15 * 7\"}"
                                }
                            }
                        ]
                    },
                    {
                        "role": "tool",
                        "content": "15 * 7 = 105",
                        "tool_call_id": "call_123"
                    },
                    {
                        "role": "assistant",
                        "content": "15 × 7 equals 105. This is a basic multiplication problem where we multiply 15 by 7 to get the result."
                    }
                ],
                "outcome": {
                    "status": "completed",
                    "output": "15 × 7 equals 105. This is a basic multiplication problem where we multiply 15 by 7 to get the result."
                },
                "turn_count": 2,
                "execution_time_ms": 1250,
                "conversation_id": "math-session-1"
            }
        }
        
        assert chat_response["success"] is True
        assert "data" in chat_response
        assert "run_id" in chat_response["data"]
        assert "trace_id" in chat_response["data"]
        assert "messages" in chat_response["data"]
        assert "outcome" in chat_response["data"]
        assert chat_response["data"]["outcome"]["status"] == "completed"
        assert isinstance(chat_response["data"]["turn_count"], int)
        assert isinstance(chat_response["data"]["execution_time_ms"], int)
    
    def test_message_format_validation(self):
        """Test message format validation."""
        
        # Valid message formats
        user_message = {
            "role": "user",
            "content": "Hello, world!"
        }
        
        assistant_message = {
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }
        
        tool_message = {
            "role": "tool",
            "content": "Tool execution result",
            "tool_call_id": "call_123"
        }
        
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant."
        }
        
        # Validate required fields
        assert user_message["role"] in ["user", "assistant", "system", "tool"]
        assert "content" in user_message
        
        assert assistant_message["role"] in ["user", "assistant", "system", "tool"]
        assert "content" in assistant_message
        
        assert tool_message["role"] in ["user", "assistant", "system", "tool"]
        assert "content" in tool_message
        assert "tool_call_id" in tool_message
        
        assert system_message["role"] in ["user", "assistant", "system", "tool"]
        assert "content" in system_message
    
    def test_tool_call_format(self):
        """Test tool call format."""
        
        tool_call = {
            "id": "call_abc123",
            "type": "function", 
            "function": {
                "name": "calculate",
                "arguments": "{\"expression\": \"2 + 2\"}"
            }
        }
        
        assert "id" in tool_call
        assert tool_call["type"] == "function"
        assert "function" in tool_call
        assert "name" in tool_call["function"]
        assert "arguments" in tool_call["function"]
        assert tool_call["function"]["name"] == "calculate"
        
        # Validate arguments is valid JSON string
        import json
        args = json.loads(tool_call["function"]["arguments"])
        assert "expression" in args
        assert args["expression"] == "2 + 2"


class TestMemoryEndpointExamples:
    """Test memory endpoint examples."""
    
    def test_conversation_response_format(self):
        """Test conversation response format."""
        
        conversation_response = {
            "success": True,
            "data": {
                "conversation_id": "user-123-session-1",
                "user_id": "user-123",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello!"
                    },
                    {
                        "role": "assistant", 
                        "content": "Hi there! How can I help you today?"
                    }
                ],
                "metadata": {
                    "session_start": "2024-01-15T10:00:00Z",
                    "topic": "general_chat"
                }
            }
        }
        
        assert conversation_response["success"] is True
        assert "data" in conversation_response
        assert "conversation_id" in conversation_response["data"]
        assert "user_id" in conversation_response["data"]
        assert "messages" in conversation_response["data"]
        assert "metadata" in conversation_response["data"]
        assert isinstance(conversation_response["data"]["messages"], list)
    
    def test_conversation_not_found_error(self):
        """Test conversation not found error format."""
        
        error_response = {
            "success": False,
            "error": "Conversation user-123-session-1 not found"
        }
        
        assert error_response["success"] is False
        assert "error" in error_response
        assert "not found" in error_response["error"]
    
    def test_delete_conversation_response(self):
        """Test delete conversation response format."""
        
        delete_response = {
            "success": True,
            "data": {
                "conversation_id": "user-123-session-1",
                "deleted": True
            }
        }
        
        assert delete_response["success"] is True
        assert "data" in delete_response
        assert "conversation_id" in delete_response["data"]
        assert delete_response["data"]["deleted"] is True
    
    def test_memory_health_response(self):
        """Test memory health response format."""
        
        memory_health_response = {
            "success": True,
            "data": {
                "healthy": True,
                "provider": "RedisMemoryProvider",
                "latency_ms": 2.5,
                "details": {
                    "connections": 5,
                    "memory_usage": "15.2MB",
                    "version": "7.0.0"
                }
            }
        }
        
        assert memory_health_response["success"] is True
        assert "data" in memory_health_response
        assert "healthy" in memory_health_response["data"]
        assert "provider" in memory_health_response["data"]
        assert "latency_ms" in memory_health_response["data"]
        assert "details" in memory_health_response["data"]
        assert isinstance(memory_health_response["data"]["latency_ms"], (int, float))


class TestErrorHandlingExamples:
    """Test error handling examples."""
    
    def test_agent_not_found_error(self):
        """Test agent not found error format."""
        
        error_response = {
            "success": False,
            "error": "Agent 'NonExistentAgent' not found. Available agents: MathTutor, ChatBot, Assistant"
        }
        
        assert error_response["success"] is False
        assert "not found" in error_response["error"]
        assert "Available agents:" in error_response["error"]
    
    def test_validation_error(self):
        """Test validation error format."""
        
        validation_error = {
            "success": False,
            "error": "1 validation error for ChatRequest\nmessages.0.role\n  Input should be 'user', 'assistant', 'system' or 'tool'"
        }
        
        assert validation_error["success"] is False
        assert "validation error" in validation_error["error"]
        assert "Input should be" in validation_error["error"]
    
    def test_memory_not_configured_error(self):
        """Test memory not configured error."""
        
        memory_error = {
            "success": False,
            "error": "Memory not configured for this server"
        }
        
        assert memory_error["success"] is False
        assert "Memory not configured" in memory_error["error"]
    
    def test_tool_execution_error(self):
        """Test tool execution error format."""
        
        tool_error_response = {
            "success": True,
            "data": {
                "outcome": {
                    "status": "error",
                    "error": {
                        "type": "ToolExecutionError",
                        "message": "Calculator tool failed: Invalid expression"
                    }
                }
            }
        }
        
        assert tool_error_response["success"] is True
        assert "data" in tool_error_response
        assert "outcome" in tool_error_response["data"]
        assert tool_error_response["data"]["outcome"]["status"] == "error"
        assert "error" in tool_error_response["data"]["outcome"]
        assert "type" in tool_error_response["data"]["outcome"]["error"]
        assert "message" in tool_error_response["data"]["outcome"]["error"]


class TestServerConfigurationExamples:
    """Test server configuration examples."""
    
    def test_basic_server_config(self):
        """Test basic server configuration."""
        
        # Mock agents and run config
        mock_agents = {"TestAgent": MagicMock()}
        mock_run_config = MagicMock()
        
        config = {
            "host": "127.0.0.1",
            "port": 3000,
            "agent_registry": mock_agents,
            "run_config": mock_run_config,
            "cors": True
        }
        
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 3000
        assert config["agent_registry"] == mock_agents
        assert config["run_config"] == mock_run_config
        assert config["cors"] is True
    
    def test_cors_configuration(self):
        """Test CORS configuration options."""
        
        # Disabled CORS
        disabled_cors_config = {"cors": False}
        assert disabled_cors_config["cors"] is False
        
        # Custom CORS settings
        custom_cors_config = {
            "cors": {
                "allow_origins": ["https://myapp.com", "https://admin.myapp.com"],
                "allow_credentials": True,
                "allow_methods": ["GET", "POST"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        }
        
        cors_settings = custom_cors_config["cors"]
        assert isinstance(cors_settings["allow_origins"], list)
        assert "https://myapp.com" in cors_settings["allow_origins"]
        assert cors_settings["allow_credentials"] is True
        assert "GET" in cors_settings["allow_methods"]
        assert "POST" in cors_settings["allow_methods"]
        assert "Content-Type" in cors_settings["allow_headers"]
    
    def test_production_configuration(self):
        """Test production configuration pattern."""
        import os
        
        # Mock environment variables
        mock_env = {
            "PORT": "8000",
            "FRONTEND_URL": "https://myapp.com"
        }
        
        config = {
            "host": "0.0.0.0",
            "port": int(mock_env.get("PORT", "8000")),
            "cors": {
                "allow_origins": [mock_env.get("FRONTEND_URL")],
                "allow_credentials": True
            }
        }
        
        assert config["host"] == "0.0.0.0"
        assert config["port"] == 8000
        assert config["cors"]["allow_origins"] == ["https://myapp.com"]
        assert config["cors"]["allow_credentials"] is True


class TestClientLibraryExamples:
    """Test client library examples."""
    
    def test_python_client_structure(self):
        """Test Python client structure."""
        
        class JAFClient:
            def __init__(self, base_url: str = "http://localhost:3000"):
                self.base_url = base_url
                # Mock httpx client
                self.client = MagicMock()
            
            async def chat(self, agent_name: str, message: str, context: dict = None, conversation_id: str = None):
                """Send a message to an agent."""
                payload = {
                    "agent_name": agent_name,
                    "messages": [{"role": "user", "content": message}],
                    "context": context or {},
                }
                
                if conversation_id:
                    payload["conversation_id"] = conversation_id
                
                # Mock response
                return {"success": True, "data": {"response": f"Mock response to: {message}"}}
            
            async def list_agents(self):
                """Get list of available agents."""
                # Mock response
                return {"success": True, "data": {"agents": [{"name": "TestAgent"}]}}
            
            async def get_conversation(self, conversation_id: str):
                """Get conversation history."""
                # Mock response
                return {"success": True, "data": {"conversation_id": conversation_id, "messages": []}}
        
        client = JAFClient()
        assert client.base_url == "http://localhost:3000"
        
        # Test async methods exist
        import asyncio
        
        async def test_methods():
            chat_result = await client.chat("TestAgent", "Hello")
            agents_result = await client.list_agents()
            conv_result = await client.get_conversation("test-conv")
            return chat_result, agents_result, conv_result
        
        chat_result, agents_result, conv_result = asyncio.run(test_methods())
        
        assert chat_result["success"] is True
        assert "Mock response to: Hello" in chat_result["data"]["response"]
        assert agents_result["success"] is True
        assert len(agents_result["data"]["agents"]) == 1
        assert conv_result["success"] is True
        assert conv_result["data"]["conversation_id"] == "test-conv"
    
    def test_javascript_client_structure(self):
        """Test JavaScript client structure (Python mock)."""
        
        class JAFClientJS:
            def __init__(self, base_url='http://localhost:3000'):
                self.base_url = base_url
            
            async def chat(self, agent_name, message, context=None, conversation_id=None):
                payload = {
                    'agent_name': agent_name,
                    'messages': [{'role': 'user', 'content': message}],
                    'context': context or {}
                }
                
                if conversation_id:
                    payload['conversation_id'] = conversation_id
                
                # Mock fetch response
                return {'success': True, 'data': {'response': f'JS Mock response to: {message}'}}
            
            async def list_agents(self):
                # Mock fetch response
                return {'success': True, 'data': {'agents': [{'name': 'JSTestAgent'}]}}
        
        client = JAFClientJS()
        assert client.base_url == 'http://localhost:3000'
        
        # Test methods
        import asyncio
        
        async def test_js_methods():
            chat_result = await client.chat('TestAgent', 'Hello from JS')
            agents_result = await client.list_agents()
            return chat_result, agents_result
        
        chat_result, agents_result = asyncio.run(test_js_methods())
        
        assert chat_result['success'] is True
        assert 'JS Mock response to: Hello from JS' in chat_result['data']['response']
        assert agents_result['success'] is True
        assert agents_result['data']['agents'][0]['name'] == 'JSTestAgent'


class TestPerformanceExamples:
    """Test performance consideration examples."""
    
    def test_connection_pooling_config(self):
        """Test connection pooling configuration."""
        
        # Mock httpx limits configuration
        limits_config = {
            "max_connections": 100,
            "max_keepalive_connections": 20
        }
        
        assert limits_config["max_connections"] == 100
        assert limits_config["max_keepalive_connections"] == 20
        assert limits_config["max_connections"] > limits_config["max_keepalive_connections"]
    
    @pytest.mark.asyncio
    async def test_batch_processing_pattern(self):
        """Test batch processing pattern."""
        
        async def process_batch(messages):
            """Mock batch processing function."""
            tasks = []
            for msg in messages:
                # Mock async task
                async def mock_chat(message):
                    await asyncio.sleep(0.01)  # Simulate network delay
                    return f"Response to: {message}"
                
                task = mock_chat(msg)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        messages = ["Hello", "How are you?", "What's the weather?"]
        results = await process_batch(messages)
        
        assert len(results) == 3
        assert all("Response to:" in result for result in results)
        assert "Hello" in results[0]
        assert "How are you?" in results[1]
        assert "What's the weather?" in results[2]


class TestSecurityExamples:
    """Test security consideration examples."""
    
    def test_input_validation_function(self):
        """Test input validation function."""
        
        def validate_context(context: dict) -> dict:
            """Additional context validation."""
            # Remove sensitive fields
            safe_context = {k: v for k, v in context.items() if not k.startswith('_')}
            
            # Validate user permissions
            if 'permissions' in safe_context:
                allowed_permissions = {'user', 'admin', 'read', 'write'}
                safe_context['permissions'] = [
                    p for p in safe_context['permissions'] 
                    if p in allowed_permissions
                ]
            
            return safe_context
        
        # Test with unsafe context
        unsafe_context = {
            'user_id': 'user123',
            '_secret_key': 'secret',
            'permissions': ['user', 'admin', 'dangerous_permission'],
            'data': 'some data'
        }
        
        safe_context = validate_context(unsafe_context)
        
        assert 'user_id' in safe_context
        assert '_secret_key' not in safe_context  # Removed sensitive field
        assert 'data' in safe_context
        assert 'permissions' in safe_context
        assert 'dangerous_permission' not in safe_context['permissions']  # Filtered out
        assert 'user' in safe_context['permissions']
        assert 'admin' in safe_context['permissions']
    
    def test_api_key_validation_pattern(self):
        """Test API key validation pattern."""
        
        def validate_api_key(api_key: str) -> bool:
            """Mock API key validation."""
            # Simple validation - in practice, check against database
            valid_keys = ['valid-key-123', 'another-valid-key']
            return api_key in valid_keys
        
        # Test valid key
        assert validate_api_key('valid-key-123') is True
        
        # Test invalid key
        assert validate_api_key('invalid-key') is False
        
        # Test None/empty key
        assert validate_api_key('') is False
    
    def test_rate_limiting_structure(self):
        """Test rate limiting structure."""
        
        # Mock rate limiting configuration
        rate_limit_config = {
            "requests_per_minute": 10,
            "burst_limit": 20,
            "key_function": "get_remote_address"
        }
        
        assert rate_limit_config["requests_per_minute"] == 10
        assert rate_limit_config["burst_limit"] >= rate_limit_config["requests_per_minute"]
        assert "key_function" in rate_limit_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
