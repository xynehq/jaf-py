"""
JAF Visualization - Type Definitions

Core types and data structures for the visualization system.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


@dataclass(frozen=True)
class NodeStyle:
    """Styling configuration for graph nodes."""
    shape: str
    fillcolor: str
    fontcolor: str
    style: str
    fontname: Optional[str] = None
    penwidth: Optional[str] = None


@dataclass(frozen=True)
class EdgeStyle:
    """Styling configuration for graph edges."""
    color: str
    style: str
    penwidth: Optional[str] = None
    arrowhead: Optional[str] = None


class GraphOptions(BaseModel):
    """Configuration options for graph generation."""
    title: Optional[str] = "JAF Graph"
    layout: Literal['dot', 'neato', 'fdp', 'circo', 'twopi'] = 'dot'
    rankdir: Literal['TB', 'LR', 'BT', 'RL'] = 'TB'
    output_format: Literal['png', 'svg', 'pdf'] = 'png'
    output_path: Optional[str] = None
    show_tool_details: bool = True
    show_sub_agents: bool = True
    color_scheme: Literal['default', 'modern', 'minimal'] = 'default'


@dataclass(frozen=True)
class GraphResult:
    """Result of graph generation operation."""
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    graph_dot: Optional[str] = None


ColorSchemeConfig = Dict[str, Dict[str, Any]]
