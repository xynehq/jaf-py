"""
Main server entry point for the JAF framework.

This module provides convenience functions for starting JAF servers
with minimal configuration.
"""

from typing import Any, Dict, List, Optional, TypeVar
from dataclasses import dataclass

from .server import JAFServer
from .types import ServerConfig
from ..core.types import Agent, RunConfig

Ctx = TypeVar('Ctx')

async def run_server(config: ServerConfig) -> None:
    """
    Start a JAF server with the given configuration.
    
    Args:
        config: Complete server configuration including agents and run config
    """
    # Create and start server
    server = JAFServer(config)
    await server.start()

def create_simple_server(
    agents: List[Agent[Any, Any]],
    model_provider: Any,
    host: str = 'localhost',
    port: int = 3000,
    max_turns: int = 10
) -> JAFServer:
    """
    Create a simple JAF server with minimal configuration.
    
    Args:
        agents: List of agents to serve
        model_provider: Model provider instance
        host: Server host (default: localhost)
        port: Server port (default: 3000)
        max_turns: Maximum turns per conversation (default: 10)
        
    Returns:
        JAFServer instance ready to start
    """
    # Create agent registry
    agent_registry = {agent.name: agent for agent in agents}
    
    # Create run config
    run_config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=max_turns
    )
    
    # Create server config
    server_config = ServerConfig(
        agent_registry=agent_registry,
        run_config=run_config,
        host=host,
        port=port,
        cors=True
    )
    
    return JAFServer(server_config)