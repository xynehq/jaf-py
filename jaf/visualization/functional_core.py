"""
JAF Visualization - Functional Core

Pure functional operations for graph creation that separate data transformation
from the imperative Graphviz operations. This follows the functional core,
imperative shell pattern.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..core.types import Agent, Tool
from .types import GraphOptions

# ========== Pure Data Structures ==========


@dataclass(frozen=True)
class NodeSpec:
    """Immutable specification for a graph node."""

    id: str
    label: str
    attributes: Dict[str, str]


@dataclass(frozen=True)
class EdgeSpec:
    """Immutable specification for a graph edge."""

    from_node: str
    to_node: str
    attributes: Dict[str, str]


@dataclass(frozen=True)
class GraphSpec:
    """Immutable specification for a complete graph."""

    title: str
    graph_attributes: Dict[str, str]
    nodes: Tuple[NodeSpec, ...]
    edges: Tuple[EdgeSpec, ...]


# ========== Pure Functions ==========


def create_graph_attributes(options: GraphOptions) -> Dict[str, str]:
    """Pure function to create graph attributes from options."""
    return {
        "rankdir": options.rankdir,
        "label": options.title or "",
        "labelloc": "t",
        "fontsize": "16",
        "fontname": "Arial Bold",
        "bgcolor": "white",
        "pad": "0.5",
    }


def create_agent_label(agent: Agent, show_tool_details: bool) -> str:
    """Pure function to create agent label."""
    model_name = getattr(agent.model_config, "name", "default") if agent.model_config else "default"
    label = f"{agent.name}\\n({model_name})"

    if show_tool_details and agent.tools:
        label += f"\\n{len(agent.tools)} tools"

    if agent.handoffs:
        label += f"\\n{len(agent.handoffs)} handoffs"

    return label


def create_tool_label(tool: Tool) -> str:
    """Pure function to create tool label."""
    description = tool.schema.description
    if len(description) > 30:
        description = description[:30] + "..."

    return f"{tool.schema.name}\\n{description}"


def create_agent_node_spec(
    agent: Agent, styles: Dict[str, Any], show_tool_details: bool
) -> NodeSpec:
    """Pure function to create agent node specification."""
    agent_style = styles["agent"]
    label = create_agent_label(agent, show_tool_details)

    attributes = {
        "label": label,
        "shape": agent_style["shape"],
        "fillcolor": agent_style["fillcolor"],
        "fontcolor": agent_style["fontcolor"],
        "style": agent_style["style"],
    }

    if "fontname" in agent_style:
        attributes["fontname"] = agent_style["fontname"]

    return NodeSpec(id=agent.name, label=label, attributes=attributes)


def create_tool_node_spec(tool: Tool, styles: Dict[str, Any]) -> NodeSpec:
    """Pure function to create tool node specification."""
    tool_style = styles["tool"]
    label = create_tool_label(tool)

    attributes = {
        "label": label,
        "shape": tool_style["shape"],
        "fillcolor": tool_style["fillcolor"],
        "fontcolor": tool_style["fontcolor"],
        "style": tool_style["style"],
    }

    if "fontname" in tool_style:
        attributes["fontname"] = tool_style["fontname"]

    return NodeSpec(id=tool.schema.name, label=label, attributes=attributes)


def create_tool_edge_spec(agent_name: str, tool_name: str, styles: Dict[str, Any]) -> EdgeSpec:
    """Pure function to create tool edge specification."""
    edge_attrs = {
        "color": styles["tool_edge"]["color"],
        "style": styles["tool_edge"]["style"],
        "penwidth": styles["tool_edge"]["penwidth"],
    }

    if "arrowhead" in styles["tool_edge"]:
        edge_attrs["arrowhead"] = styles["tool_edge"]["arrowhead"]

    return EdgeSpec(from_node=agent_name, to_node=tool_name, attributes=edge_attrs)


def create_handoff_edge_spec(from_agent: str, to_agent: str, styles: Dict[str, Any]) -> EdgeSpec:
    """Pure function to create handoff edge specification."""
    return EdgeSpec(
        from_node=from_agent,
        to_node=to_agent,
        attributes={"color": styles["edge"]["color"], "style": "dashed", "label": "handoff"},
    )


def create_agent_graph_spec(
    agents: List[Agent], options: GraphOptions, styles: Dict[str, Any]
) -> GraphSpec:
    """Pure function to create complete agent graph specification."""
    nodes = []
    edges = []

    # Create agent nodes and their tool nodes/edges
    for agent in agents:
        # Add agent node
        agent_node = create_agent_node_spec(agent, styles, options.show_tool_details)
        nodes.append(agent_node)

        # Add tool nodes and edges if requested
        if options.show_tool_details and agent.tools:
            for tool in agent.tools:
                tool_node = create_tool_node_spec(tool, styles)
                nodes.append(tool_node)

                tool_edge = create_tool_edge_spec(agent.name, tool.schema.name, styles)
                edges.append(tool_edge)

        # Add handoff edges
        if agent.handoffs:
            for handoff_target in agent.handoffs:
                handoff_edge = create_handoff_edge_spec(agent.name, handoff_target, styles)
                edges.append(handoff_edge)

    return GraphSpec(
        title=options.title or "JAF Graph",
        graph_attributes=create_graph_attributes(options),
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def create_tool_graph_spec(
    tools: List[Tool], options: GraphOptions, styles: Dict[str, Any]
) -> GraphSpec:
    """Pure function to create tool graph specification."""
    nodes = []

    for tool in tools:
        tool_node = create_tool_node_spec(tool, styles)
        nodes.append(tool_node)

    return GraphSpec(
        title=options.title or "JAF Tool Graph",
        graph_attributes=create_graph_attributes(options),
        nodes=tuple(nodes),
        edges=tuple(),  # No edges in pure tool graph
    )
