#!/usr/bin/env python3
"""
A2A Implementation Validation Script

This script validates that the complete A2A implementation is working correctly
by testing core functionality without external dependencies.

Validates:
- Type system and Pydantic models
- Agent creation and transformation
- Protocol handlers
- Client-server interaction patterns
- Example functionality

Usage:
    python validate_a2a_implementation.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Test core imports
    from jaf.a2a import (
        # Core functionality
        A2A, create_a2a_agent, create_a2a_tool,
        create_server_config, create_a2a_client,
        
        # Types
        A2AMessage, A2ATask, A2AAgent, A2AAgentTool,
        create_a2a_message, create_a2a_text_part,
        
        # Protocol
        validate_jsonrpc_request, create_jsonrpc_success_response_dict,
        
        # Constants
        A2A_PROTOCOL_VERSION, A2A_SUPPORTED_METHODS
    )
    
    print("✅ Core A2A imports successful")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


async def test_agent_creation():
    """Test agent creation functionality"""
    print("\n🤖 Testing agent creation...")
    
    try:
        # Create a simple tool
        async def test_tool(args, context):
            return {"result": f"Tool executed with: {args}"}
        
        tool = create_a2a_tool(
            "test_tool",
            "A test tool",
            {"type": "object", "properties": {"input": {"type": "string"}}},
            test_tool
        )
        
        # Create agent
        agent = create_a2a_agent(
            "TestAgent",
            "A test agent for validation",
            "You are a helpful test agent",
            [tool]
        )
        
        # Validate agent properties
        assert agent.name == "TestAgent"
        assert agent.description == "A test agent for validation"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "test_tool"
        
        print("✅ Agent creation successful")
        return agent
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        raise


def test_type_system():
    """Test type system and Pydantic models"""
    print("\n📋 Testing type system...")
    
    try:
        # Test message creation
        message = create_a2a_message(
            role="user",
            parts=[create_a2a_text_part("Hello, A2A world!")],
            context_id="test_context"
        )
        
        # Validate message
        assert message.role == "user"
        assert message.context_id == "test_context"
        assert len(message.parts) == 1
        assert message.parts[0].text == "Hello, A2A world!"
        
        # Test serialization
        message_dict = message.model_dump()
        assert isinstance(message_dict, dict)
        
        # Test JSON serialization
        json_str = json.dumps(message_dict)
        assert isinstance(json_str, str)
        
        print("✅ Type system validation successful")
        
    except Exception as e:
        print(f"❌ Type system validation failed: {e}")
        raise


def test_protocol_validation():
    """Test protocol validation functions"""
    print("\n🔍 Testing protocol validation...")
    
    try:
        # Test valid JSON-RPC request
        valid_request = {
            "jsonrpc": "2.0",
            "id": "test_123",
            "method": "message/send",
            "params": {"test": "data"}
        }
        
        assert validate_jsonrpc_request(valid_request) is True
        
        # Test invalid request
        invalid_request = {
            "id": "test_123",
            "method": "message/send"
            # Missing jsonrpc field
        }
        
        assert validate_jsonrpc_request(invalid_request) is False
        
        # Test response creation
        response = create_jsonrpc_success_response_dict(
            "test_123",
            {"status": "success"}
        )
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test_123"
        assert response["result"]["status"] == "success"
        
        print("✅ Protocol validation successful")
        
    except Exception as e:
        print(f"❌ Protocol validation failed: {e}")
        raise


def test_server_configuration():
    """Test server configuration creation"""
    print("\n🖥️ Testing server configuration...")
    
    try:
        # Create test agent
        agent = create_a2a_agent(
            "ServerTestAgent",
            "Agent for server testing",
            "You are a server test agent",
            []
        )
        
        # Create server configuration
        config = create_server_config(
            agents={"ServerTestAgent": agent},
            name="Validation Test Server",
            description="Server for A2A implementation validation",
            port=3000,
            host="localhost"
        )
        
        # Validate configuration
        assert config["agentCard"]["name"] == "Validation Test Server"
        assert config["port"] == 3000
        assert config["host"] == "localhost"
        assert "ServerTestAgent" in config["agents"]
        
        print("✅ Server configuration successful")
        return config
        
    except Exception as e:
        print(f"❌ Server configuration failed: {e}")
        raise


def test_client_creation():
    """Test client creation"""
    print("\n📱 Testing client creation...")
    
    try:
        # Create client
        client = create_a2a_client("http://localhost:3000")
        
        # Validate client
        assert client.config.base_url == "http://localhost:3000"
        assert client.config.timeout == 30000
        assert client.session_id.startswith("client_")
        
        print("✅ Client creation successful")
        return client
        
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        raise


def test_convenience_api():
    """Test convenience API functionality"""
    print("\n🎯 Testing convenience API...")
    
    try:
        # Test A2A convenience class
        agent = A2A.agent("ConvenienceAgent", "Test agent", "You are helpful")
        assert agent.name == "ConvenienceAgent"
        
        client = A2A.client("http://test.example.com")
        assert client.config.base_url == "http://test.example.com"
        
        async def test_tool_func(args, context):
            return {"result": "convenience test"}
        
        tool = A2A.tool("conv_tool", "Convenience tool", {}, test_tool_func)
        assert tool.name == "conv_tool"
        
        server_config = A2A.server(
            {"ConvenienceAgent": agent},
            "Convenience Server",
            "Test server",
            3001
        )
        assert server_config["agentCard"]["name"] == "Convenience Server"
        
        print("✅ Convenience API successful")
        
    except Exception as e:
        print(f"❌ Convenience API failed: {e}")
        raise


def test_constants_and_metadata():
    """Test protocol constants and metadata"""
    print("\n📊 Testing constants and metadata...")
    
    try:
        # Test protocol version
        assert A2A_PROTOCOL_VERSION == "0.3.0"
        
        # Test supported methods
        expected_methods = [
            "message/send",
            "message/stream",
            "tasks/get", 
            "tasks/cancel",
            "agent/getAuthenticatedExtendedCard"
        ]
        
        for method in expected_methods:
            assert method in A2A_SUPPORTED_METHODS
        
        print("✅ Constants and metadata validation successful")
        
    except Exception as e:
        print(f"❌ Constants validation failed: {e}")
        raise


async def test_tool_execution():
    """Test tool execution"""
    print("\n🔧 Testing tool execution...")
    
    try:
        # Create and execute tool
        async def math_tool(args, context):
            a = args.get("a", 0)
            b = args.get("b", 0)
            return {"result": a + b, "operation": "addition"}
        
        tool = create_a2a_tool(
            "math_add",
            "Addition tool",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            },
            math_tool
        )
        
        # Execute tool
        result = await tool.execute({"a": 25, "b": 17}, {})
        
        assert result["result"] == 42
        assert result["operation"] == "addition"
        
        print("✅ Tool execution successful")
        
    except Exception as e:
        print(f"❌ Tool execution failed: {e}")
        raise


async def main():
    """Main validation function"""
    print("🚀 Starting A2A Implementation Validation")
    print("=" * 50)
    
    try:
        # Run all validation tests
        test_type_system()
        test_protocol_validation()
        await test_agent_creation()
        test_server_configuration()
        test_client_creation()
        test_convenience_api()
        test_constants_and_metadata()
        await test_tool_execution()
        
        print("\n" + "=" * 50)
        print("🎉 A2A Implementation Validation SUCCESSFUL!")
        print("\n✅ All core components are working correctly:")
        print("   • Type system and Pydantic models")
        print("   • Agent creation and tools")
        print("   • Protocol validation")
        print("   • Server configuration")
        print("   • Client creation")
        print("   • Convenience API")
        print("   • Constants and metadata")
        print("   • Tool execution")
        
        print("\n🔥 The A2A implementation is ready for use!")
        print("\n📚 Next steps:")
        print("   • Run examples: python jaf/a2a/examples/integration_example.py")
        print("   • Run tests: python jaf/a2a/tests/run_tests.py")
        print("   • Start server: python jaf/a2a/examples/server_example.py")
        print("   • Connect client: python jaf/a2a/examples/client_example.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        print("\n🔧 Please check the implementation and try again.")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)