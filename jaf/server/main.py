"""
Main server entry point for the JAF framework.

This module provides convenience functions for starting JAF servers
with minimal configuration.
"""

from typing import Dict, List, Optional, TypeVar, Union

import uvicorn

from ..core.types import Agent, RunConfig
from ..memory.types import MemoryProvider
from .server import create_jaf_server
from .types import ServerConfig

Ctx = TypeVar("Ctx")


async def run_server(
    agents: Union[List[Agent], Dict[str, Agent]],
    run_config: RunConfig,
    host: str = "127.0.0.1",
    port: int = 3000,
    cors: bool = True,
    default_memory_provider: Optional[MemoryProvider] = None,
) -> None:
    """
    Create and start a JAF server with the given configuration.

    Args:
        agents: List or dictionary of agents to serve.
        run_config: Core run configuration.
        host: Server host.
        port: Server port.
        cors: Enable/disable CORS.
        default_memory_provider: Optional default memory provider for the server.
    """
    if isinstance(agents, list):
        agent_registry = {agent.name: agent for agent in agents}
    else:
        agent_registry = agents

    if not agent_registry:
        raise ValueError("At least one agent must be provided.")

    # Create server config
    server_config = ServerConfig(
        agent_registry=agent_registry,
        run_config=run_config,
        host=host,
        port=port,
        cors=cors,
        default_memory_provider=default_memory_provider,
    )

    app = create_jaf_server(server_config)

    # Configure and run uvicorn
    uv_config = uvicorn.Config(app=app, host=host, port=port, log_level="info")

    server = uvicorn.Server(uv_config)

    print(f"🚀 JAF Server running on http://{host}:{port}")
    print(f"📋 Available agents: {', '.join(agent_registry.keys())}")
    print(f"📖 API docs: http://{host}:{port}/docs")
    if default_memory_provider:
        print(f"🧠 Memory provider: {type(default_memory_provider).__name__}")
    else:
        print("⚠️  Memory: Not configured (conversations will not persist)")

    await server.serve()
