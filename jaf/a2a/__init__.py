"""
JAF A2A (Agent-to-Agent) Communication Protocol

This module provides a complete implementation of the A2A protocol for JAF,
enabling distributed agent communication through JSON-RPC over HTTP.

Main exports:
- Types: All A2A protocol types and models
- Client: HTTP client for A2A communication
- Server: FastAPI server with A2A endpoints
- Agent: A2A agent creation and transformation utilities
- Protocol: JSON-RPC protocol handlers
- AgentCard: Dynamic agent discovery

Usage Examples:

# Quick client connection:
client = await connect_to_a2a_agent("http://localhost:3000")
response = await client["ask"]("Hello, how can you help me?")

# Create A2A agent:
agent = create_a2a_agent("MyAgent", "A helpful agent", "You are helpful", tools)

# Start A2A server:
server_config = create_server_config(agents, "My Server", "Multi-agent server", 3000)
server = await start_a2a_server(server_config)
"""

# Core A2A Types
from .types import (
    # Core A2A protocol types
    A2AMessage, A2ATask, A2AAgent, A2AAgentTool, A2AArtifact,
    A2ATaskStatus, A2APart, A2AStreamEvent, A2AError, A2AErrorCodes,
    
    # JSON-RPC types
    JSONRPCRequest, JSONRPCResponse, JSONRPCError,
    SendMessageRequest, SendStreamingMessageRequest, GetTaskRequest,
    
    # Agent and configuration types
    AgentCard, AgentSkill, AgentCapabilities, AgentProvider,
    A2AServerConfig, A2AClientConfig, A2AClientState,
    MessageSendConfiguration, ToolContext, A2AToolResult,
    
    # Stream and state types
    AgentState, StreamEvent, TaskState,
    
    # Factory functions
    create_a2a_message, create_a2a_task, create_a2a_agent_tool,
    create_a2a_artifact, create_a2a_text_part, create_a2a_data_part,
    create_jsonrpc_request, create_jsonrpc_success_response,
    create_jsonrpc_error_response, create_a2a_error
)

# Agent utilities
from .agent import (
    # Agent creation
    create_a2a_agent, create_a2a_tool,
    
    # Agent transformation
    transform_a2a_agent_to_jaf, transform_a2a_tool_to_jaf,
    
    # State management
    create_initial_agent_state, add_message_to_state,
    update_state_from_run_result, create_user_message,
    
    # Message processing
    process_agent_query, extract_text_from_a2a_message,
    create_a2a_text_message, create_a2a_data_message,
    
    # Task management
    create_a2a_task, update_a2a_task_status, add_artifact_to_a2a_task,
    complete_a2a_task,
    
    # Execution functions
    execute_a2a_agent, execute_a2a_agent_with_streaming,
    
    # Configuration
    create_run_config_for_a2a_agent, transform_to_run_state
)

# Protocol handlers
from .protocol import (
    # Validation
    validate_jsonrpc_request, validate_send_message_request,
    
    # Request handlers
    handle_message_send, handle_message_stream,
    handle_tasks_get, handle_tasks_cancel,
    handle_get_authenticated_extended_card,
    
    # Response utilities
    create_jsonrpc_success_response_dict, create_jsonrpc_error_response_dict,
    map_error_to_a2a_error,
    
    # Routing
    route_a2a_request, create_protocol_handler_config
)

# Agent Card generation
from .agent_card import (
    # Agent card generation
    generate_agent_card, generate_agent_card_for_agent,
    create_minimal_agent_card, create_agent_card_from_config,
    
    # Skills generation
    generate_skills_from_agents, generate_examples_for_agent,
    generate_examples_for_tool,
    
    # Security
    generate_security_schemes, generate_security_requirements,
    
    # Validation
    validate_agent_card, is_valid_url,
    
    # Utilities
    merge_agent_cards
)

# Server functionality
from .server import (
    # Server creation
    create_a2a_server, create_fastapi_app, start_a2a_server,
    
    # Configuration
    create_a2a_server_config, create_server_config,
    
    # Route setup
    setup_a2a_routes,
    
    # Request handling
    handle_a2a_request_internal, handle_a2a_request_for_agent,
    route_a2a_request_wrapper
)

# Client functionality
from .client import (
    # Client creation
    create_a2a_client, connect_to_a2a_agent,
    
    # Message handling
    send_message, stream_message, send_message_to_agent, stream_message_to_agent,
    
    # Request creation
    create_message_request, create_streaming_message_request,
    
    # HTTP utilities
    send_http_request, send_a2a_request,
    
    # Discovery
    get_agent_card, discover_agents,
    
    # Health and capabilities
    check_a2a_health, get_a2a_capabilities,
    
    # Response processing
    extract_text_response,
    
    # Utilities
    create_a2a_message_dict, parse_sse_event, validate_a2a_response
)

# Main exports for easy access
__all__ = [
    # Types
    "A2AMessage", "A2ATask", "A2AAgent", "A2AAgentTool", "A2AArtifact",
    "A2ATaskStatus", "A2APart", "A2AStreamEvent", "A2AError", "A2AErrorCodes",
    "JSONRPCRequest", "JSONRPCResponse", "JSONRPCError",
    "AgentCard", "AgentSkill", "AgentCapabilities", "AgentProvider",
    "A2AServerConfig", "A2AClientConfig", "A2AClientState",
    
    # Agent utilities
    "create_a2a_agent", "create_a2a_tool", "transform_a2a_agent_to_jaf",
    "process_agent_query", "execute_a2a_agent", "execute_a2a_agent_with_streaming",
    
    # Protocol
    "validate_jsonrpc_request", "route_a2a_request",
    
    # Agent Card
    "generate_agent_card", "validate_agent_card",
    
    # Server
    "create_a2a_server", "start_a2a_server", "create_server_config",
    
    # Client
    "create_a2a_client", "connect_to_a2a_agent", "send_message",
    "stream_message", "discover_agents"
]


# Convenience A2A class for common operations
class A2A:
    """
    Convenience class for A2A operations
    
    Usage:
        # Create client
        a2a = A2A.client("http://localhost:3000")
        response = await a2a.ask("Hello")
        
        # Create server
        server = A2A.server(agents, "My Server", "Description", 3000)
        await server.start()
    """
    
    @staticmethod
    def client(base_url: str, config=None):
        """Create A2A client with convenient methods"""
        return create_a2a_client(base_url, config)
    
    @staticmethod
    async def connect(base_url: str):
        """Connect to A2A agent with full capabilities"""
        return await connect_to_a2a_agent(base_url)
    
    @staticmethod
    def agent(name: str, description: str, instruction: str, tools=None):
        """Create A2A agent"""
        return create_a2a_agent(name, description, instruction, tools or [])
    
    @staticmethod
    def server(agents: dict, name: str, description: str, port: int, **kwargs):
        """Create A2A server configuration"""
        return create_server_config(agents, name, description, port, **kwargs)
    
    @staticmethod
    async def start_server(config: dict):
        """Start A2A server"""
        return await start_a2a_server(config)
    
    @staticmethod
    def tool(name: str, description: str, parameters: dict, execute_func):
        """Create A2A tool"""
        return create_a2a_tool(name, description, parameters, execute_func)


# Version info
__version__ = "1.0.0"
__protocol_version__ = "0.3.0"
__author__ = "JAF Framework"
__description__ = "Agent-to-Agent Communication Protocol for JAF"

# Protocol information
A2A_PROTOCOL_VERSION = "0.3.0"
A2A_SUPPORTED_METHODS = [
    "message/send",
    "message/stream", 
    "tasks/get",
    "tasks/cancel",
    "agent/getAuthenticatedExtendedCard"
]
A2A_SUPPORTED_TRANSPORTS = ["JSONRPC"]
A2A_DEFAULT_CAPABILITIES = {
    "streaming": True,
    "pushNotifications": False,
    "stateTransitionHistory": True
}