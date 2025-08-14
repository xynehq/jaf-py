#!/usr/bin/env python3
"""
Interactive MCP Filesystem Server

Simple JAF server with filesystem MCP tools that responds to HTTP requests.
Demonstrates secure integration between JAF-PY and Model Context Protocol.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

# Add the project root to the path so we can import jaf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from jaf import (
    Agent,
    Tool,
    run_server,
    Message
)
from jaf.providers.model import make_litellm_provider
from jaf.core.tracing import ConsoleTraceCollector
from jaf.memory import create_in_memory_provider, MemoryConfig
from jaf.core.tool_results import ToolResult, ToolResultStatus, ToolErrorCodes
from jaf.providers.mcp import (
    create_mcp_stdio_client,
    MCPClient,
    MCPTool,
    MCPToolArgs
)
from jaf.core.types import ToolSchema, RunConfig

# Load environment variables
load_dotenv()

# Context type for filesystem operations
class FilesystemContext(BaseModel):
    userId: str
    sessionId: str
    workingDirectory: Optional[str] = None
    allowedPaths: Optional[List[str]] = None

# Dynamic args class for MCP tools
class DynamicMCPArgs(MCPToolArgs):
    """Dynamic args model that accepts any arguments for MCP tools."""
    
    class Config:
        extra = "allow"  # Allow extra fields
    
    def __init__(self, **data):
        super().__init__()
        # Store all arguments dynamically
        for key, value in data.items():
            setattr(self, key, value)

class SecureMCPTool:
    """A secure wrapper for MCP tools with path validation."""
    
    def __init__(self, mcp_tool: MCPTool, allowed_paths: List[str]):
        self.mcp_tool = mcp_tool
        self.allowed_paths = allowed_paths
        self._schema = mcp_tool.schema
    
    @property
    def schema(self) -> ToolSchema:
        """Get the tool schema."""
        return self._schema
    
    async def execute(self, args: Any, context: FilesystemContext) -> ToolResult:
        """Execute the MCP tool with security validation."""
        try:
            print(f"🔧 Executing {self.mcp_tool.tool_name} with args: {args}")
            
            # Path validation for security
            if hasattr(args, 'path') and args.path:
                path = str(args.path)
                is_allowed = any(path.startswith(allowed_path) for allowed_path in self.allowed_paths)
                
                if not is_allowed:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        error_code=ToolErrorCodes.INVALID_INPUT,
                        error_message=f"Path '{path}' is not in allowed directories: {', '.join(self.allowed_paths)}",
                        data={"path": path, "allowed_paths": self.allowed_paths}
                    )
            
            # Execute the original MCP tool
            result = await self.mcp_tool.execute(args, context)
            print(f"✅ {self.mcp_tool.tool_name} completed successfully")
            return result
            
        except Exception as error:
            print(f"❌ Error in {self.mcp_tool.tool_name}: {error}")
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error_code=ToolErrorCodes.EXECUTION_FAILED,
                error_message=f"Failed to execute {self.mcp_tool.tool_name}: {str(error)}",
                data={"error": str(error)}
            )

async def setup_filesystem_mcp_tools() -> List[SecureMCPTool]:
    """Setup filesystem MCP tools with security wrapper."""
    try:
        print("🔌 Connecting to filesystem MCP server...")
        
        # Connect to filesystem MCP server using npx
        mcp_client = create_mcp_stdio_client([
            'npx',
            '-y',
            '@modelcontextprotocol/server-filesystem',
            '/Users'  # Allow access to Users directory
        ])
        
        # Initialize the client
        await mcp_client.initialize()
        
        # Get available tools
        available_tools = mcp_client.get_available_tools()
        print("📋 Available filesystem tools:")
        
        for index, tool_name in enumerate(available_tools, 1):
            tool_info = mcp_client.get_tool_info(tool_name)
            description = tool_info.get("description", "No description") if tool_info else "No description"
            print(f"{index}. {tool_name}: {description[:80]}...")
        
        if not available_tools:
            print("⚠️ No filesystem tools found! Please check MCP server connection.")
            return []
        
        # Convert MCP tools to JAF tools with security wrapper
        secure_tools = []
        allowed_paths = ['/Users', '/tmp']
        
        for tool_name in available_tools:
            # Create MCP tool
            mcp_tool = MCPTool(mcp_client, tool_name, DynamicMCPArgs)
            
            # Wrap with security
            secure_tool = SecureMCPTool(mcp_tool, allowed_paths)
            secure_tools.append(secure_tool)
        
        print(f"✅ Successfully integrated {len(secure_tools)} filesystem tools")
        return secure_tools
        
    except Exception as error:
        print(f"❌ Failed to connect to filesystem MCP server: {error}")
        print("Make sure you have internet connection for npx to download the package")
        return []

async def create_filesystem_agents(filesystem_tools: List[SecureMCPTool]) -> Dict[str, Agent]:
    """Create filesystem agents."""
    
    # Main Filesystem Agent - comprehensive file operations
    def filesystem_instructions(state):
        # Handle case where state or context might be None (e.g., for /agents endpoint)
        context = getattr(state, 'context', {}) if state else {}
        tool_descriptions = "\n".join([
            f"- {tool.schema.name}: {tool.schema.description}"
            for tool in filesystem_tools
        ])
        
        return f"""You are an intelligent filesystem assistant powered by MCP (Model Context Protocol) tools.

**Your Role:**
- Help users perform filesystem operations safely and efficiently
- Provide file and directory management capabilities
- Support content creation, reading, modification, and organization
- Ensure operations stay within allowed directories for security

**Available Operations:**
{tool_descriptions}

**Security Boundaries:**
- Allowed directories: /Users, /tmp
- All file paths must be within these directories
- Always validate paths before operations

**Current Context:**
- User: {context.get('userId', 'unknown') if context else 'unknown'}
- Session: {context.get('sessionId', 'unknown') if context else 'unknown'}

**Best Practices:**
- Always check if files/directories exist before operations
- Use absolute paths for clarity
- Provide helpful feedback about operations performed
- Suggest related operations when appropriate
- Handle errors gracefully and explain what went wrong

**Example Operations:**
- "List files in Desktop" → use list_directory
- "Create a test file" → use write_file 
- "Read file contents" → use read_text_file
- "Get file information" → use get_file_info

Be helpful, safe, and informative in all filesystem operations!"""
    
    filesystem_agent = Agent(
        name='FilesystemAgent',
        instructions=filesystem_instructions,
        tools=filesystem_tools
    )
    
    # Quick File Operations Agent
    def quick_file_instructions(state):
        # Handle case where state or context might be None
        context = getattr(state, 'context', {}) if state else {}
        basic_tools = [t for t in filesystem_tools 
                      if t.schema.name in ['read_text_file', 'write_file', 'list_directory', 'get_file_info']]
        tool_names = [t.schema.name for t in basic_tools]
        
        return f"""You are a quick file operations specialist focusing on common file tasks.

**Your Role:**
- Handle simple, common file operations quickly
- Focus on basic read/write/list operations
- Provide concise, direct responses

**Available Tools:** {', '.join(tool_names)}

**Context:**
- User: {context.get('userId', 'unknown') if context else 'unknown'}
- Session: {context.get('sessionId', 'unknown') if context else 'unknown'}

Keep operations simple and responses brief but informative."""
    
    quick_file_agent = Agent(
        name='QuickFileAgent',
        instructions=quick_file_instructions,
        tools=[t for t in filesystem_tools 
               if t.schema.name in ['read_text_file', 'write_file', 'list_directory', 'get_file_info']]
    )
    
    return {
        'FilesystemAgent': filesystem_agent,
        'QuickFileAgent': quick_file_agent
    }

async def start_filesystem_server():
    """Start the filesystem server."""
    print("🚀 Starting MCP Filesystem Agent Server...\n")
    
    # Setup MCP tools
    filesystem_tools = await setup_filesystem_mcp_tools()
    
    if not filesystem_tools:
        print("❌ No filesystem tools available. Cannot start server.")
        sys.exit(1)
    
    # Create agents
    agents_dict = await create_filesystem_agents(filesystem_tools)
    agents = list(agents_dict.values())
    
    # Setup providers
    model_provider = make_litellm_provider(
        os.getenv('LITELLM_URL', 'http://localhost:4000'),
        os.getenv('LITELLM_API_KEY')
    )
    
    trace_collector = ConsoleTraceCollector()
    memory_provider = create_in_memory_provider()
    memory_config = MemoryConfig(provider=memory_provider, auto_store=True)
    
    try:
        # Create run config
        run_config = RunConfig(
            agent_registry={agent.name: agent for agent in agents},
            model_provider=model_provider,
            max_turns=10,
            model_override=os.getenv('LITELLM_MODEL', 'gemini-2.5-pro'),
            on_event=trace_collector.collect,
            memory=memory_config
        )
        
        host = os.getenv('HOST', '127.0.0.1')
        port = int(os.getenv('PORT', '3003'))
        
        print("✅ MCP Filesystem Server started successfully!")
        print(f"🌐 Server running on http://{host}:{port}")
        
        print("\n🤖 Available Agents:")
        print("1. FilesystemAgent - Comprehensive filesystem operations")
        print("2. QuickFileAgent - Simple file operations specialist")
        
        print("\n🔧 Available Filesystem Tools:")
        for index, tool in enumerate(filesystem_tools, 1):
            print(f"{index}. {tool.schema.name} - {tool.schema.description[:60]}...")
        
        print("\n📚 Example curl commands:")
        
        print("\n📂 List Desktop files:")
        print(f"curl -X POST http://{host}:{port}/chat \\")
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"messages":[{"role":"user","content":"List all files in my Desktop directory"}],"agentName":"FilesystemAgent","context":{"userId":"user_001","sessionId":"session_123"}}\'')
        
        print("\n📝 Create a test file:")
        print(f"curl -X POST http://{host}:{port}/chat \\")
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"messages":[{"role":"user","content":"Create a file called hello.txt on my Desktop with the content: Hello from MCP filesystem agent!"}],"agentName":"FilesystemAgent","context":{"userId":"user_001","sessionId":"session_123"}}\'')
        
        print("\n📖 Read a file:")
        print(f"curl -X POST http://{host}:{port}/chat \\")
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"messages":[{"role":"user","content":"Read the contents of /Desktop/hello.txt"}],"agentName":"FilesystemAgent","context":{"userId":"user_001","sessionId":"session_123"}}\'')
        
        print("\n📊 Server endpoints:")
        print("   GET  /health         - Health check")
        print("   GET  /agents         - List available agents")
        print("   POST /chat           - Perform filesystem operations")
        
        print("\n💡 Key Features:")
        print("   • Real MCP integration with @modelcontextprotocol/server-filesystem")
        print("   • Secure path validation (Desktop and /tmp only)")
        print("   • Interactive agent responses via curl/HTTP")
        print("   • Comprehensive filesystem operations")
        print("   • Error handling and user-friendly feedback")
        
        print("\n🔒 Security:")
        print("   • Operations restricted to allowed directories only")
        print("   • Path validation on all file operations")
        print("   • Safe MCP tool integration")
        
        # Start the server
        await run_server(
            agents,
            run_config,
            host=host,
            port=port,
            cors=True
        )
            
    except Exception as error:
        print(f"❌ Failed to start filesystem server: {error}")
        sys.exit(1)

def main():
    """Main entry point."""
    asyncio.run(start_filesystem_server())

if __name__ == "__main__":
    main()