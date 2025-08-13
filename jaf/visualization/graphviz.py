"""
JAF Visualization - Graphviz Integration

Functional visualization system for agents and tools using Graphviz.
Provides the core functionality to generate visual representations of agent
architectures, tool ecosystems, and system relationships.
"""

# ========== Color Schemes (Immutable Constants) ==========
from typing import Any, Dict, Final, List, Optional

from graphviz import Digraph

from ..core.types import Agent, Tool
from .types import GraphOptions, GraphResult

COLOR_SCHEMES: Final[Dict[str, Dict[str, Any]]] = {
    'default': {
        'agent': {
            'shape': 'box',
            'fillcolor': '#E3F2FD',
            'fontcolor': '#1976D2',
            'style': 'filled,rounded'
        },
        'tool': {
            'shape': 'ellipse',
            'fillcolor': '#F3E5F5',
            'fontcolor': '#7B1FA2',
            'style': 'filled'
        },
        'sub_agent': {
            'shape': 'box',
            'fillcolor': '#E8F5E8',
            'fontcolor': '#388E3C',
            'style': 'filled,dashed'
        },
        'edge': {
            'color': '#424242',
            'style': 'solid',
            'penwidth': '1.5'
        },
        'tool_edge': {
            'color': '#9C27B0',
            'style': 'dashed',
            'penwidth': '1.0'
        }
    },
    'modern': {
        'agent': {
            'shape': 'box',
            'fillcolor': '#667eea',
            'fontcolor': 'white',
            'style': 'filled,rounded',
            'fontname': 'Arial Bold'
        },
        'tool': {
            'shape': 'ellipse',
            'fillcolor': '#f093fb',
            'fontcolor': 'white',
            'style': 'filled',
            'fontname': 'Arial'
        },
        'sub_agent': {
            'shape': 'box',
            'fillcolor': '#4facfe',
            'fontcolor': 'white',
            'style': 'filled,dashed',
            'fontname': 'Arial'
        },
        'edge': {
            'color': '#667eea',
            'style': 'solid',
            'penwidth': '2.0',
            'arrowhead': 'vee'
        },
        'tool_edge': {
            'color': '#f093fb',
            'style': 'dashed',
            'penwidth': '1.5',
            'arrowhead': 'open'
        }
    },
    'minimal': {
        'agent': {
            'shape': 'box',
            'fillcolor': 'white',
            'fontcolor': 'black',
            'style': 'filled',
            'penwidth': '2'
        },
        'tool': {
            'shape': 'ellipse',
            'fillcolor': '#f5f5f5',
            'fontcolor': 'black',
            'style': 'filled'
        },
        'sub_agent': {
            'shape': 'box',
            'fillcolor': 'white',
            'fontcolor': 'gray',
            'style': 'filled,dashed'
        },
        'edge': {
            'color': 'black',
            'style': 'solid',
            'penwidth': '1.0'
        },
        'tool_edge': {
            'color': 'gray',
            'style': 'dashed',
            'penwidth': '1.0'
        }
    }
}


# ========== Core Graph Generation Functions ==========

async def generate_agent_graph(
    agents: List[Agent],
    options: Optional[GraphOptions] = None
) -> GraphResult:
    """
    Generate a visual graph of agent architecture using functional approach.
    
    Args:
        agents: List of Agent objects to visualize
        options: Graph generation options
        
    Returns:
        GraphResult with success status and output information
    """
    from .functional_core import create_agent_graph_spec
    from .imperative_shell import render_graph_spec

    try:
        opts = options or GraphOptions()

        # Validate options (pure function)
        errors = validate_graph_options(opts)
        if errors:
            return GraphResult(
                success=False,
                error=f"Invalid options: {', '.join(errors)}"
            )

        # Get color scheme (immutable data)
        styles = COLOR_SCHEMES[opts.color_scheme]

        # Create pure graph specification (functional core)
        graph_spec = create_agent_graph_spec(agents, opts, styles)

        # Render graph using imperative shell
        return render_graph_spec(graph_spec, opts, 'AgentGraph')

    except Exception as error:
        return GraphResult(
            success=False,
            error=str(error)
        )


async def generate_tool_graph(
    tools: List[Tool],
    options: Optional[GraphOptions] = None
) -> GraphResult:
    """
    Generate a visual graph of tool ecosystem using functional approach.
    
    Args:
        tools: List of Tool objects to visualize
        options: Graph generation options
        
    Returns:
        GraphResult with success status and output information
    """
    from .functional_core import create_tool_graph_spec
    from .imperative_shell import render_graph_spec

    try:
        opts = options or GraphOptions(
            title="JAF Tool Graph",
            layout='circo'
        )

        # Validate options (pure function)
        errors = validate_graph_options(opts)
        if errors:
            return GraphResult(
                success=False,
                error=f"Invalid options: {', '.join(errors)}"
            )

        # Get color scheme (immutable data)
        styles = COLOR_SCHEMES[opts.color_scheme]

        # Create pure graph specification (functional core)
        graph_spec = create_tool_graph_spec(tools, opts, styles)

        # Render graph using imperative shell
        return render_graph_spec(graph_spec, opts, 'ToolGraph')

    except Exception as error:
        return GraphResult(
            success=False,
            error=str(error)
        )


async def generate_runner_graph(
    agent_registry: Dict[str, Agent],
    options: Optional[GraphOptions] = None
) -> GraphResult:
    """
    Generate a visual graph of runner architecture.
    
    Args:
        agent_registry: Dictionary of agent name to Agent object
        options: Graph generation options
        
    Returns:
        GraphResult with success status and output information
    """
    try:
        opts = options or GraphOptions(
            title="JAF Runner Architecture",
            color_scheme='modern'
        )

        # Validate options
        errors = validate_graph_options(opts)
        if errors:
            return GraphResult(
                success=False,
                error=f"Invalid options: {', '.join(errors)}"
            )

        # Create digraph
        graph = Digraph('RunnerGraph', comment=opts.title)
        graph.attr('graph', compound='true')

        # Set graph attributes
        graph.attr('graph',
                  rankdir=opts.rankdir,
                  label=opts.title or '',
                  labelloc='t',
                  fontsize='16',
                  fontname='Arial Bold',
                  bgcolor='white',
                  pad='0.5')

        # Get color scheme
        styles = COLOR_SCHEMES[opts.color_scheme]

        # Create clusters for different components
        with graph.subgraph(name='cluster_agents') as agent_cluster:
            agent_cluster.attr(label='Agents')
            agent_cluster.attr(style='filled')
            agent_cluster.attr(fillcolor='#f8f9fa')

            # Add agents to cluster
            for agent_name, agent in agent_registry.items():
                # Create agent label
                model_name = getattr(agent.model_config, 'name', 'default') if agent.model_config else 'default'
                label = f"{agent.name}\\n({model_name})"

                if opts.show_tool_details and agent.tools:
                    label += f"\\n{len(agent.tools)} tools"

                if agent.handoffs:
                    label += f"\\n{len(agent.handoffs)} handoffs"

                # Add agent node
                agent_cluster.node(
                    agent.name,
                    label=label,
                    shape=styles['agent']['shape'],
                    fillcolor=styles['agent']['fillcolor'],
                    fontcolor=styles['agent']['fontcolor'],
                    style=styles['agent']['style']
                )

        with graph.subgraph(name='cluster_session') as session_cluster:
            session_cluster.attr(label='Session Layer')
            session_cluster.attr(style='filled')
            session_cluster.attr(fillcolor='#fff3cd')

            # Add session provider
            session_cluster.node(
                'session_provider',
                label='Session\\nProvider',
                shape='box',
                fillcolor='#ffc107',
                fontcolor='black',
                style='filled,rounded'
            )

        # Add runner node
        graph.node(
            'runner',
            label='Runner',
            shape='diamond',
            fillcolor='#28a745',
            fontcolor='white',
            style='filled',
            fontsize='14',
            fontname='Arial Bold'
        )

        # Add edges
        for agent_name in agent_registry:
            graph.edge('runner', agent_name, color=styles['edge']['color'],
                      style=styles['edge']['style'], label='executes')

        graph.edge('runner', 'session_provider', color='#ffc107',
                  style='dashed', label='manages')

        # Generate output
        output_path = opts.output_path or f"./runner-graph.{opts.output_format}"

        # Render the graph
        graph.render(
            filename=output_path.replace(f'.{opts.output_format}', ''),
            format=opts.output_format,
            cleanup=True
        )

        return GraphResult(
            success=True,
            output_path=output_path,
            graph_dot=graph.source
        )

    except Exception as error:
        return GraphResult(
            success=False,
            error=str(error)
        )


# ========== Helper Functions (Deprecated - moved to functional_core) ==========
# These functions have been moved to functional_core.py for better functional design


# ========== Validation Functions ==========

def validate_graph_options(options: GraphOptions) -> List[str]:
    """
    Validate graph options and return list of errors.
    
    Args:
        options: GraphOptions to validate
        
    Returns:
        List of validation error messages
    """
    errors = []

    valid_layouts = ['dot', 'neato', 'fdp', 'circo', 'twopi']
    if options.layout not in valid_layouts:
        errors.append(f"Invalid layout '{options.layout}'. Must be one of: {valid_layouts}")

    valid_rankdirs = ['TB', 'LR', 'BT', 'RL']
    if options.rankdir not in valid_rankdirs:
        errors.append(f"Invalid rankdir '{options.rankdir}'. Must be one of: {valid_rankdirs}")

    valid_formats = ['png', 'svg', 'pdf']
    if options.output_format not in valid_formats:
        errors.append(f"Invalid output_format '{options.output_format}'. Must be one of: {valid_formats}")

    valid_schemes = ['default', 'modern', 'minimal']
    if options.color_scheme not in valid_schemes:
        errors.append(f"Invalid color_scheme '{options.color_scheme}'. Must be one of: {valid_schemes}")

    return errors


# ========== Utility Functions ==========

def get_graph_dot(agents: List[Agent], options: Optional[GraphOptions] = None) -> str:
    """
    Get the DOT language representation of an agent graph using functional approach.
    
    Args:
        agents: List of Agent objects
        options: Graph generation options
        
    Returns:
        DOT language string representation
    """
    from .functional_core import create_agent_graph_spec
    from .imperative_shell import graph_spec_to_dot

    opts = options or GraphOptions()

    # Get color scheme (immutable data)
    styles = COLOR_SCHEMES[opts.color_scheme]

    # Create pure graph specification (functional core)
    graph_spec = create_agent_graph_spec(agents, opts, styles)

    # Convert to DOT using imperative shell (no file system side effects)
    return graph_spec_to_dot(graph_spec, 'AgentGraph')
