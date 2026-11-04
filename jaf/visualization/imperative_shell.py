"""
JAF Visualization - Imperative Shell

This module contains the imperative operations that interact with the external
Graphviz library. It uses the pure functional core to generate specifications
and then applies them to create actual graphs.

This follows the functional core, imperative shell pattern where:
- All business logic is pure and functional (in functional_core.py)
- All side effects are contained here in the imperative shell
"""

import os
from graphviz import Digraph

from .functional_core import GraphSpec
from .types import GraphOptions, GraphResult


def apply_graph_spec_to_digraph(spec: GraphSpec, digraph: Digraph) -> Digraph:
    """
    Apply a pure GraphSpec to a mutable Digraph object.
    This is the imperative shell that contains all side effects.
    """
    # Apply graph attributes
    digraph.attr("graph", **spec.graph_attributes)

    # Add all nodes
    for node in spec.nodes:
        digraph.node(node.id, **node.attributes)

    # Add all edges
    for edge in spec.edges:
        digraph.edge(edge.from_node, edge.to_node, **edge.attributes)

    return digraph


def render_graph_spec(
    spec: GraphSpec, options: GraphOptions, graph_name: str = "Graph"
) -> GraphResult:
    """
    Render a GraphSpec to an actual file using Graphviz.
    This encapsulates all file system side effects.
    """
    try:
        # Create fresh Digraph
        graph = Digraph(graph_name, comment=spec.title)

        # Apply the pure specification
        apply_graph_spec_to_digraph(spec, graph)

        # Handle output path
        output_path = options.output_path or f"./{graph_name.lower()}.{options.output_format}"

        # Never write outside the folder the caller pointed at via ".." segments
        if ".." in os.path.normpath(output_path).split(os.sep):
            return GraphResult(success=False, error="Output path must not contain '..'")

        # Render (side effect); strip only a trailing extension, not every occurrence
        suffix = f".{options.output_format}"
        render_name = output_path[: -len(suffix)] if output_path.endswith(suffix) else output_path
        graph.render(
            filename=render_name,
            format=options.output_format,
            cleanup=True,
        )

        return GraphResult(success=True, output_path=output_path, graph_dot=graph.source)

    except Exception as error:
        return GraphResult(success=False, error=str(error))


def graph_spec_to_dot(spec: GraphSpec, graph_name: str = "Graph") -> str:
    """
    Convert a GraphSpec to DOT language string without file system side effects.
    This is a pure operation that only creates the DOT representation.
    """
    # Create fresh Digraph
    graph = Digraph(graph_name, comment=spec.title)

    # Apply the pure specification
    apply_graph_spec_to_digraph(spec, graph)

    # Return DOT source (no side effects)
    return graph.source
