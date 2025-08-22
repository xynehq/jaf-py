"""
JAF (Juspay Agent Framework) - Python Implementation

A purely functional agent framework with immutable state and composable tools.
"""

from .core.engine import run
from .core.errors import JAFError as _LegacyJAFError
from .core.tool_results import *
from .core.tools import (
    create_function_tool,
    create_function_tool_legacy,
    create_async_function_tool,
    create_async_function_tool_legacy,
    function_tool,
)
from .core.tracing import ConsoleTraceCollector, TraceCollector
from .core.types import *
from .exceptions import (
    A2AException,
    A2AProtocolError,
    A2ATaskError,
    AgentException,
    AgentNotFoundError,
    ConfigurationException,
    GuardrailViolationError,
    HandoffError,
    InputValidationError,
    InvalidConfigurationError,
    JAFError,
    JAFException,
    MaxTurnsExceededError,
    MemoryConnectionError,
    MemoryException,
    MemoryStorageError,
    ModelException,
    ModelProviderError,
    ModelResponseError,
    SessionException,
    SessionStateError,
    ToolException,
    ToolExecutionError,
    ToolValidationError,
    ValidationException,
    create_agent_error,
    create_memory_error,
    create_session_error,
    create_tool_error,
)
from .memory import *
from .policies.handoff import *
from .policies.validation import *
from .providers.mcp import (
    FastMCPTool,
    MCPToolArgs,
    create_mcp_stdio_tools,
    create_mcp_sse_tools,
    create_mcp_http_tools,
)
from .providers.model import make_litellm_provider
from .server import run_server

# New features (conditional imports)
try:
    from .core.performance import (
        PerformanceMonitor,
        PerformanceMetrics,
        monitor_performance,
        get_performance_summary,
    )
    _PERFORMANCE_AVAILABLE = True
except ImportError:
    _PERFORMANCE_AVAILABLE = False

try:
    from .core.streaming import (
        run_streaming,
        StreamingEvent,
        StreamingEventType,
        StreamingCollector,
        create_sse_response,
        stream_to_websocket,
    )
    _STREAMING_AVAILABLE = True
except ImportError:
    _STREAMING_AVAILABLE = False

try:
    from .core.composition import (
        create_tool_pipeline,
        create_parallel_tools,
        create_conditional_tool,
        with_retry,
        with_cache,
        with_timeout,
        compose,
    )
    _COMPOSITION_AVAILABLE = True
except ImportError:
    _COMPOSITION_AVAILABLE = False

try:
    from .plugins import (
        JAFPlugin,
        PluginMetadata,
        PluginStatus,
        PluginRegistry,
        get_plugin_registry,
    )
    _PLUGINS_AVAILABLE = True
except ImportError:
    _PLUGINS_AVAILABLE = False

try:
    from .core.analytics import (
        ConversationAnalytics,
        AgentAnalytics,
        SystemAnalytics,
        AnalyticsEngine,
        get_analytics_report,
        analyze_conversation_quality,
    )
    _ANALYTICS_AVAILABLE = True
except ImportError:
    _ANALYTICS_AVAILABLE = False

try:
    from .core.workflows import (
        Workflow,
        WorkflowStep,
        WorkflowContext,
        WorkflowResult,
        StepResult,
        WorkflowStatus,
        StepStatus,
        AgentStep,
        ToolStep,
        ConditionalStep,
        ParallelStep,
        LoopStep,
        WorkflowBuilder,
        create_workflow,
        create_sequential_workflow,
        create_parallel_workflow,
        execute_workflow_stream,
    )
    _WORKFLOWS_AVAILABLE = True
except ImportError:
    _WORKFLOWS_AVAILABLE = False

# Visualization (optional import)
try:
    from .visualization import (
        GraphOptions,
        GraphResult,
        generate_agent_graph,
        generate_runner_graph,
        generate_tool_graph,
        get_graph_dot,
        validate_graph_options,
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

import uuid

from .core.types import RunId, TraceId, create_run_id, create_trace_id


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
    
    # Enums for type safety
    "Model", "ToolParameterType", "ToolSource", "ContentRole", "PartType",
    
    # Tool factory functions
    "create_function_tool", "create_function_tool_legacy",
    "create_async_function_tool", "create_async_function_tool_legacy",
    "FunctionToolConfig", "ToolExecuteFunction", "function_tool",

    # Exception classes
    "JAFException", "AgentException", "AgentNotFoundError", "HandoffError",
    "ToolException", "ToolExecutionError", "ToolValidationError",
    "ModelException", "ModelProviderError", "ModelResponseError",
    "ValidationException", "GuardrailViolationError", "InputValidationError",
    "MemoryException", "MemoryConnectionError", "MemoryStorageError",
    "SessionException", "SessionStateError", "MaxTurnsExceededError",
    "A2AException", "A2AProtocolError", "A2ATaskError",
    "ConfigurationException", "InvalidConfigurationError",
    "create_agent_error", "create_tool_error", "create_session_error", "create_memory_error",

    # Engine
    "run",

    # Tracing
    "TraceCollector", "ConsoleTraceCollector",

    # Tool results
    "ToolResult", "ToolResultStatus", "ToolResponse", "ToolErrorCodes",
    "with_error_handling", "require_permissions", "tool_result_to_string",

    # Providers
    "make_litellm_provider",
    "FastMCPTool",
    "MCPToolArgs",
    "create_mcp_stdio_tools",
    "create_mcp_sse_tools",
    "create_mcp_http_tools",

    # Memory system
    "ConversationMemory", "MemoryProvider", "MemoryQuery", "MemoryConfig",
    "Result", "Success", "Failure",
    "InMemoryConfig", "RedisConfig", "PostgresConfig", "MemoryProviderConfig",
    "MemoryError", "MemoryConnectionError", "MemoryNotFoundError", "MemoryStorageError",
    "create_memory_provider_from_env", "get_memory_provider_info", "test_memory_provider_connection",
    "create_in_memory_provider", "create_redis_provider", "create_postgres_provider",

    # Server
    "run_server",
    
    # Performance monitoring (NEW)
    "PerformanceMonitor", "PerformanceMetrics", "monitor_performance", "get_performance_summary",
    
    # Streaming support (NEW)
    "run_streaming", "StreamingEvent", "StreamingEventType", "StreamingCollector",
    "create_sse_response", "stream_to_websocket",
    
    # Tool composition (NEW)
    "create_tool_pipeline", "create_parallel_tools", "create_conditional_tool",
    "with_retry", "with_cache", "with_timeout", "compose",
    
    # Plugin system (NEW)
    "JAFPlugin", "PluginMetadata", "PluginStatus", "PluginRegistry", "get_plugin_registry",
    
    # Analytics system (NEW)
    "ConversationAnalytics", "AgentAnalytics", "SystemAnalytics", "AnalyticsEngine",
    "get_analytics_report", "analyze_conversation_quality",
    
    # Workflow orchestration (NEW)
    "Workflow", "WorkflowStep", "WorkflowContext", "WorkflowResult", "StepResult",
    "WorkflowStatus", "StepStatus", "AgentStep", "ToolStep", "ConditionalStep",
    "ParallelStep", "LoopStep", "WorkflowBuilder", "create_workflow",
    "create_sequential_workflow", "create_parallel_workflow", "execute_workflow_stream",
] + (
    # Visualization (conditional)
    [
        "generate_agent_graph", "generate_tool_graph", "generate_runner_graph",
        "GraphOptions", "GraphResult", "get_graph_dot", "validate_graph_options"
    ] if _VISUALIZATION_AVAILABLE else []
)
