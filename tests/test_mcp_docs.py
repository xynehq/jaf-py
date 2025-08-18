#!/usr/bin/env python3
"""
Test script to validate MCP documentation examples.
"""

import asyncio
import sys
import os
from typing import List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from jaf.providers.mcp import (
    create_mcp_stdio_client,
    create_mcp_websocket_client,
    create_mcp_sse_client,
    create_mcp_http_client,
    MCPTool,
    MCPToolArgs,
    create_mcp_tools_from_client
)
from jaf.core.tool_results import ToolResult, ToolResultStatus
from pydantic import BaseModel

class DynamicMCPArgs(MCPToolArgs):
    """Dynamic args that accept any parameters."""
    class Config:
        extra = "allow"
    
    def __init__(self, **data):
        super().__init__()
        for key, value in data.items():
            setattr(self, key, value)

class FileReadArgs(MCPToolArgs):
    path: str

async def test_stdio_transport():
    """Test stdio transport with filesystem server."""
    print("🧪 Testing stdio transport...")
    
    try:
        # Test the example from docs
        mcp_client = create_mcp_stdio_client([
            'npx', '-y', '@modelcontextprotocol/server-filesystem', '/tmp'
        ])
        
        await mcp_client.initialize()
        print("✅ Stdio transport connection successful")
        
        # Test tool discovery
        tools = mcp_client.get_available_tools()
        print(f"✅ Found {len(tools)} tools: {tools}")
        
        # Test creating tools
        if tools:
            # Test with specific args
            if 'read_text_file' in tools:
                mcp_tool = MCPTool(mcp_client, "read_text_file", FileReadArgs)
                print("✅ Created MCPTool with specific args")
            
            # Test with dynamic args
            first_tool = tools[0]
            dynamic_tool = MCPTool(mcp_client, first_tool, DynamicMCPArgs)
            print("✅ Created MCPTool with dynamic args")
        
        # Test create_mcp_tools_from_client function
        auto_tools = await create_mcp_tools_from_client(mcp_client)
        print(f"✅ Auto-created {len(auto_tools)} tools")
        
        await mcp_client.close()
        print("✅ Stdio transport test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Stdio transport test failed: {e}")
        return False

async def test_websocket_transport():
    """Test websocket transport (will fail without server)."""
    print("🧪 Testing websocket transport...")
    
    try:
        # This will fail without a running WebSocket server, but tests the API
        mcp_client = create_mcp_websocket_client('ws://localhost:8080/mcp')
        print("✅ WebSocket client created successfully")
        
        # Don't initialize as there's no server
        print("✅ WebSocket transport API test completed")
        return True
        
    except Exception as e:
        print(f"❌ WebSocket transport test failed: {e}")
        return False

async def test_sse_transport():
    """Test SSE transport (will fail without server)."""
    print("🧪 Testing SSE transport...")
    
    try:
        # This will fail without a running SSE server, but tests the API
        mcp_client = create_mcp_sse_client('http://localhost:8080/events')
        print("✅ SSE client created successfully")
        
        # Don't initialize as there's no server
        print("✅ SSE transport API test completed")
        return True
        
    except Exception as e:
        print(f"❌ SSE transport test failed: {e}")
        return False

async def test_http_transport():
    """Test HTTP transport (will fail without server)."""
    print("🧪 Testing HTTP transport...")
    
    try:
        # This will fail without a running HTTP server, but tests the API
        mcp_client = create_mcp_http_client('http://localhost:8080/mcp')
        print("✅ HTTP client created successfully")
        
        # Don't initialize as there's no server
        print("✅ HTTP transport API test completed")
        return True
        
    except Exception as e:
        print(f"❌ HTTP transport test failed: {e}")
        return False

async def test_documentation_examples():
    """Test key examples from the documentation."""
    print("🧪 Testing documentation examples...")
    
    results = []
    
    # Test 1: Basic client creation (from docs)
    try:
        # From docs: create_mcp_stdio_client
        mcp_client = create_mcp_stdio_client([
            'npx', '-y', '@modelcontextprotocol/server-filesystem', '/tmp'
        ])
        print("✅ Example 1: create_mcp_stdio_client works")
        results.append(True)
    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        results.append(False)
    
    # Test 2: WebSocket client creation
    try:
        mcp_client = create_mcp_websocket_client('ws://localhost:8080/mcp')
        print("✅ Example 2: create_mcp_websocket_client works")
        results.append(True)
    except Exception as e:
        print(f"❌ Example 2 failed: {e}")
        results.append(False)
    
    # Test 3: SSE client creation
    try:
        mcp_client = create_mcp_sse_client('http://localhost:8080/events')
        print("✅ Example 3: create_mcp_sse_client works")
        results.append(True)
    except Exception as e:
        print(f"❌ Example 3 failed: {e}")
        results.append(False)
    
    # Test 4: HTTP client creation
    try:
        mcp_client = create_mcp_http_client('http://localhost:8080/mcp')
        print("✅ Example 4: create_mcp_http_client works")
        results.append(True)
    except Exception as e:
        print(f"❌ Example 4 failed: {e}")
        results.append(False)
    
    return all(results)

async def main():
    """Run all tests."""
    print("🚀 Starting MCP documentation validation tests...\n")
    
    test_results = []
    
    # Test documentation examples
    result = await test_documentation_examples()
    test_results.append(result)
    print()
    
    # Test stdio transport (requires internet for npx)
    result = await test_stdio_transport()
    test_results.append(result)
    print()
    
    # Test other transports (API only)
    result = await test_websocket_transport()
    test_results.append(result)
    print()
    
    result = await test_sse_transport()
    test_results.append(result)
    print()
    
    result = await test_http_transport()
    test_results.append(result)
    print()
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All MCP documentation examples are working correctly!")
        return True
    else:
        print("❌ Some MCP documentation examples have issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
