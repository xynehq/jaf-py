#!/usr/bin/env python3

"""
HITL Demo Server - HTTP server for JAF HITL functionality

This module provides a server showcasing JAF's Human-in-the-Loop 
capabilities with approval-requiring tools.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jaf.server.server import create_jaf_server
from jaf.server.types import ServerConfig
from jaf.providers.model import make_litellm_provider
from jaf.core.types import RunConfig, Agent, Tool, ToolSchema
from jaf.memory.approval_storage import create_in_memory_approval_storage
from pydantic import BaseModel

from shared.memory import setup_memory_provider, Colors


class RedirectParams(BaseModel):
    url: str
    reason: str = None


class SendDataParams(BaseModel):
    data: str
    recipient: str


def create_redirect_tool() -> Tool[RedirectParams, Any]:
    """Tool that requires approval - redirect user to different screen/page."""
    
    class RedirectTool:
        @property
        def schema(self) -> ToolSchema[RedirectParams]:
            return ToolSchema(
                name='redirectUser',
                description='Redirect user to a different screen/page',
                parameters=RedirectParams
            )
        
        @property
        def needs_approval(self) -> bool:
            return True
        
        async def execute(self, args: RedirectParams, context: Any) -> str:
            # Simulate using context provided through approval
            prev = f" from {context.get('currentScreen', '')}" if context.get('currentScreen') else ""
            reason = args.reason or "n/a"
            return f"Redirected user{prev} to {args.url}. Reason: {reason}"
    
    return RedirectTool()


def create_send_data_tool() -> Tool[SendDataParams, Any]:
    """Tool that requires approval - send sensitive data."""
    
    class SendDataTool:
        @property
        def schema(self) -> ToolSchema[SendDataParams]:
            return ToolSchema(
                name='sendSensitiveData', 
                description='Send sensitive data to a recipient',
                parameters=SendDataParams
            )
        
        @property
        def needs_approval(self) -> bool:
            return True
        
        async def execute(self, args: SendDataParams, context: Any) -> str:
            level = context.get('encryptionLevel', 'none')
            return f"Sent data to {args.recipient} with encryption={level}."
    
    return SendDataTool()


def create_hitl_agent() -> Agent[Any, str]:
    """Create the HITL demo agent."""
    return Agent(
        name='HITL Demo Agent',
        instructions=lambda state: """You are a helpful assistant. Use tools when appropriate.
Tools:
- redirectUser (requires approval)
- sendSensitiveData (requires approval)
""",
        tools=[create_redirect_tool(), create_send_data_tool()],
        model_config={'name': os.getenv('LITELLM_MODEL', 'gpt-3.5-turbo'), 'temperature': 0.1}
    )


def create_model_provider():
    """Create model provider with environment configuration."""
    base_url = os.getenv('LITELLM_URL', 'http://localhost:4000')
    api_key = os.getenv('LITELLM_API_KEY', 'sk-demo')
    
    print(Colors.green(f'🤖 Using LiteLLM: {base_url}'))
    return make_litellm_provider(base_url, api_key)


async def main():
    """Main server function."""
    
    # Configuration
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '3000'))
    
    # Model provider
    model_provider = create_model_provider()
    
    # Memory provider from env
    memory_provider = await setup_memory_provider()
    
    # Approval storage
    approval_storage = create_in_memory_approval_storage()
    
    # Create agent
    hitl_agent = create_hitl_agent()
    agent_registry = {'HITL Demo Agent': hitl_agent}
    
    # Run configuration
    from jaf.memory.types import MemoryConfig
    memory_config = MemoryConfig(
        provider=memory_provider,
        auto_store=True,
        max_messages=200
    )
    
    run_config = RunConfig(
        agent_registry=agent_registry,
        model_provider=model_provider,
        max_turns=6,
        memory=memory_config,
        approval_storage=approval_storage
    )
    
    # Server configuration
    server_config = ServerConfig(
        agent_registry=agent_registry,
        run_config=run_config,
        default_memory_provider=memory_provider,
        cors=True
    )
    
    # Create server
    app = create_jaf_server(server_config)
    
    # Usage hints
    print(Colors.green('\\n✅ HITL Server Running'))
    print(f'Base URL: http://{host}:{port}')
    print()
    
    print('Endpoints:')
    print(f'• Health:               GET  /health')
    print(f'• Agents:               GET  /agents')
    print(f'• Chat:                 POST /chat')
    print(f'• Pending Approvals:    GET  /approvals/pending?conversationId=...')
    print(f'• Approvals SSE Stream: GET  /approvals/stream?conversationId=...')
    print()
    
    # Start server
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Colors.cyan('\\n👋 Server stopped'))
    except Exception as e:
        print(Colors.yellow(f'Error: {e}'))
        import traceback
        traceback.print_exc()