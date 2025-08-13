"""
JAF (Juspay Agent Framework) - Python Implementation

A purely functional agent framework with immutable state and composable tools.
"""

from .core.types import *
from .core.engine import run
from .core.tracing import TraceCollector, ConsoleTraceCollector
from .core.errors import JAFError
from .core.tool_results import *

from .providers.model import make_litellm_provider
from .providers.mcp import (
    MCPClient, MCPTool, MCPToolArgs,
    create_mcp_websocket_client, create_mcp_stdio_client,
    create_mcp_tools_from_client
)

from .policies.validation import *
from .policies.handoff import *

from .memory import *

from .server import run_server

# Visualization (optional import)
try:
    from .visualization import (
        generate_agent_graph, generate_tool_graph, generate_runner_graph,
        GraphOptions, GraphResult, get_graph_dot, validate_graph_options
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

import uuid
from .core.types import TraceId, RunId, create_trace_id, create_run_id

def generate_trace_id() -> TraceId:
    """Generate a new trace ID."""
    return create_trace_id(str(uuid.uuid4()))

def generate_run_id() -> RunId:
    """Generate a new run ID."""
    return create_run_id(str(uuid.uuid4()))

__version__ = "2.0.0"
__all__ = [
    # Core types and functions
    "TraceId", "RunId", "ValidationResult", "Message", "ModelConfig", 
    "Tool", "Agent", "Guardrail", "RunState", "JAFError", "RunResult", 
    "TraceEvent", "ModelProvider", "RunConfig",
    "create_trace_id", "create_run_id", "generate_trace_id", "generate_run_id",
    
    # Engine
    "run",
    
    # Tracing
    "TraceCollector", "ConsoleTraceCollector",
    
    # Tool results
    "ToolResult", "ToolResultStatus", "ToolResponse", "ToolErrorCodes",
    "with_error_handling", "require_permissions", "tool_result_to_string",
    
    # Providers
    "make_litellm_provider",
    "MCPClient", "MCPTool", "MCPToolArgs",
    "create_mcp_websocket_client", "create_mcp_stdio_client",
    "create_mcp_tools_from_client",
    
    # Memory system
    "ConversationMemory", "MemoryProvider", "MemoryQuery", "MemoryConfig",
    "Result", "Success", "Failure",
    "InMemoryConfig", "RedisConfig", "PostgresConfig", "MemoryProviderConfig",
    "MemoryError", "MemoryConnectionError", "MemoryNotFoundError", "MemoryStorageError",
    "create_memory_provider_from_env", "get_memory_provider_info", "test_memory_provider_connection",
    "create_in_memory_provider", "create_redis_provider", "create_postgres_provider",
    
    # Server
    "run_server",
] + (
    # Visualization (conditional)
    [
        "generate_agent_graph", "generate_tool_graph", "generate_runner_graph",
        "GraphOptions", "GraphResult", "get_graph_dot", "validate_graph_options"
    ] if _VISUALIZATION_AVAILABLE else []
)