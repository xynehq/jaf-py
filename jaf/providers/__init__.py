"""JAF Providers module - Model providers and external integrations."""

from .mcp import (
    FastMCPTool,
    MCPToolArgs,
    create_mcp_stdio_tools,
    create_mcp_sse_tools,
    create_mcp_http_tools,
)
from .model import make_litellm_provider

# Back-compat for renamed/removed exports (do not add to __all__)
import warnings as _warnings

_DEPRECATED_ALIASES = {
    "MCPTool": (FastMCPTool, "FastMCPTool"),
    "create_mcp_stdio_client": (create_mcp_stdio_tools, "create_mcp_stdio_tools"),
    "create_mcp_sse_client": (create_mcp_sse_tools, "create_mcp_sse_tools"),
}

_REMOVED_EXPORTS = {
    # No safe automatic migration known — force an explicit choice.
    "MCPClient": "MCPClient was removed. Use transport-specific tool factories: "
                 "create_mcp_stdio_tools, create_mcp_sse_tools, or create_mcp_http_tools.",
    "create_mcp_tools_from_client": "create_mcp_tools_from_client was removed. "
                                    "Construct tools via create_mcp_stdio_tools, "
                                    "create_mcp_sse_tools, or create_mcp_http_tools as appropriate.",
}

def __getattr__(name: str):
    if name in _DEPRECATED_ALIASES:
        obj, new_name = _DEPRECATED_ALIASES[name]
        _warnings.warn(
            f"jaf.providers.{name} is deprecated; use jaf.providers.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj
    if name in _REMOVED_EXPORTS:
        _warnings.warn(
            f"jaf.providers.{name} is no longer available: {_REMOVED_EXPORTS[name]}",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import-time attribute access should fail clearly.
        raise AttributeError(f"{name} has been removed")
    raise AttributeError(name)

def __dir__():
    # Make deprecated names discoverable in REPLs without advertising in __all__
    return sorted(set(globals()) | set(_DEPRECATED_ALIASES))

__all__ = [
    "FastMCPTool",
    "MCPToolArgs",
    "create_mcp_stdio_tools",
    "create_mcp_sse_tools",
    "create_mcp_http_tools",
    "make_litellm_provider",
]
