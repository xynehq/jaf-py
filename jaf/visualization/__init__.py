"""
JAF Visualization Module

Provides architectural visualization capabilities for agent systems using Graphviz.
Allows developers to generate visual diagrams of their agent architectures, tools,
and system relationships.

Core functionality:
- Agent architecture visualization
- Tool ecosystem diagrams  
- Runner system visualization
- Multiple output formats (PNG, SVG, PDF)
- Customizable color schemes and layouts
"""

from .graphviz import (
    generate_agent_graph,
    generate_tool_graph,
    generate_runner_graph,
    get_graph_dot,
    validate_graph_options,
)
from .types import (
    GraphOptions,
    GraphResult,
    NodeStyle,
    EdgeStyle,
)

try:
    from .example import (
        run_visualization_examples,
        quick_start_visualization,
    )
except ImportError:
    # Example module may not be available in all environments
    pass

__all__ = [
    # Core functions
    "generate_agent_graph",
    "generate_tool_graph", 
    "generate_runner_graph",
    "get_graph_dot",
    "validate_graph_options",
    # Types
    "GraphOptions",
    "GraphResult",
    "NodeStyle", 
    "EdgeStyle",
    # Examples (if available)
    "run_visualization_examples",
    "quick_start_visualization",
]