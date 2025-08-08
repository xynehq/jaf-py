"""
JAF (Juspay Agent Framework) - Python Implementation

A purely functional agent framework with immutable state and composable tools.
"""

from .core.types import *
from .core.engine import run
from .core.tracing import TraceCollector, ConsoleTraceCollector
from .core.errors import FAFError
from .core.tool_results import *

from .providers.model import make_litellm_provider
# from .providers.mcp import *  # Commented out for test compatibility

from .policies.validation import *
from .policies.handoff import *

from .server import run_server

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
    "Tool", "Agent", "Guardrail", "RunState", "FAFError", "RunResult", 
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
    
    # Server
    "run_server",
]