#!/usr/bin/env python3

"""
File System HITL JAF Server Demo

This demo showcases the file system HITL functionality using JAF server:
- All file operations from the main demo
- JAF server with standard HTTP API endpoints
- Built-in approval management via JAF endpoints
- Real-time coordination between clients and server

Usage: python examples/hitl-demo/api_demo.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn

from jaf.server.server import create_jaf_server
from jaf.server.types import ServerConfig
from jaf.core.types import RunConfig
from jaf.providers.model import make_litellm_provider

from shared.agent import file_system_agent, LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL
from shared.tools import FileSystemContext, DEMO_DIR
from shared.memory import setup_memory_provider, Colors


# Configuration
API_PORT = int(os.getenv('API_PORT', '3001'))


def create_model_provider():
    """Create model provider - requires LiteLLM configuration."""
    # Check if we have environment variables set (not using defaults)
    has_env_config = os.getenv('LITELLM_BASE_URL') or os.getenv('LITELLM_URL')
    has_api_key = os.getenv('LITELLM_API_KEY')

    if not has_env_config or not has_api_key:
        print(Colors.yellow('❌ No LiteLLM configuration found'))
        print(Colors.yellow('   Please set LITELLM_BASE_URL and LITELLM_API_KEY environment variables'))
        print(Colors.yellow('   Example: LITELLM_BASE_URL=http://localhost:4000 LITELLM_API_KEY=your-key python examples/hitl-demo/api_demo.py'))
        print(Colors.dim('   Or copy examples/hitl-demo/.env.example to .env and configure your LiteLLM server'))
        sys.exit(1)

    print(Colors.green(f'🤖 Using LiteLLM: {LITELLM_BASE_URL} ({LITELLM_MODEL})'))
    return make_litellm_provider(LITELLM_BASE_URL, LITELLM_API_KEY)


def setup_sandbox():
    """Setup demo sandbox directory."""
    try:
        DEMO_DIR.mkdir(parents=True, exist_ok=True)

        demo_files = [
            {
                'name': 'README.txt',
                'content': 'Welcome to the File System HITL JAF Server Demo!\nThis is a sample file for testing.'
            },
            {
                'name': 'config.json',
                'content': '{\n  "app": "filesystem-jaf-server-demo",\n  "version": "1.0.0",\n  "server": "JAF"\n}'
            },
            {
                'name': 'notes.md',
                'content': '# JAF Server Demo Notes\n\n- This is a markdown file\n- You can edit or delete it via JAF endpoints\n- Operations require approval'
            }
        ]

        for file_info in demo_files:
            file_path = DEMO_DIR / file_info['name']
            if not file_path.exists():
                file_path.write_text(file_info['content'], encoding='utf-8')

        print(Colors.green(f'📁 Sandbox directory ready: {DEMO_DIR}'))

    except Exception as e:
        print(Colors.yellow(f'Failed to setup sandbox: {e}'))
        sys.exit(1)


def display_welcome():
    """Display welcome message."""
    os.system('clear' if os.name == 'posix' else 'cls')
    print(Colors.cyan('🌐 JAF File System HITL Server Demo'))
    print(Colors.cyan('==================================='))
    print()

    print(Colors.green('This demo showcases HITL with JAF server endpoints:'))
    print(Colors.green('• Safe operations: listFiles, readFile (no approval)'))
    print(Colors.green('• Dangerous operations: deleteFile, editFile (require approval)'))
    print(Colors.green('• Chat via JAF server endpoints'))
    print(Colors.green('• Approval management via JAF server endpoints'))
    print(Colors.green('• Integrated approval storage in memory provider'))
    print()

    print(Colors.cyan('Example requests:'))
    print('• "list files in the current directory"')
    print('• "read the README file"')
    print('• "edit the config file to add server: JAF"')
    print('• "delete the notes file"')
    print()

    print(Colors.yellow('JAF Server Endpoints:'))
    print(f'• Health:               GET  http://localhost:{API_PORT}/health')
    print(f'• Agents:               GET  http://localhost:{API_PORT}/agents')
    print(f'• Chat:                 POST http://localhost:{API_PORT}/chat')
    print(f'• Pending Approvals:    GET  http://localhost:{API_PORT}/approvals/pending?conversationId=...')
    print(f'• Approvals SSE Stream: GET  http://localhost:{API_PORT}/approvals/stream?conversationId=...')
    print()

    print(Colors.dim('Use the JAF server endpoints to interact with the agent'))
    print()


def setup_api_server(config: RunConfig[FileSystemContext]):
    """Setup JAF HTTP API server."""

    # Create agent registry
    agent_registry = {'FileSystemAgent': file_system_agent}

    # Server configuration
    server_config = ServerConfig(
        agent_registry=agent_registry,
        run_config=config,
        default_memory_provider=config.memory.provider,
        cors=True
    )

    # Create JAF server
    app = create_jaf_server(server_config)

    return app


async def main():
    """Main demo function."""
    display_welcome()
    setup_sandbox()

    # Setup memory provider (now includes approval storage automatically)
    memory_provider = await setup_memory_provider()

    from jaf.memory.types import MemoryConfig
    memory_config = MemoryConfig(
        provider=memory_provider,
        auto_store=True,
        max_messages=50,
        store_on_completion=True
    )

    config = RunConfig(
        agent_registry={'FileSystemAgent': file_system_agent},
        model_provider=create_model_provider(),
        memory=memory_config,
        conversation_id=f'filesystem-jaf-server-demo-{int(time.time() * 1000)}'
    )

    # Setup JAF API server
    app = setup_api_server(config)

    # Generate session ID for this demo run
    session_id = f"jaf-server-demo-{int(time.time() * 1000)}"
    print(Colors.cyan(f'🔗 Session ID: {session_id}'))
    print()

    print(Colors.green(f'🌐 JAF server running on http://localhost:{API_PORT}'))
    print(Colors.dim(f'   Health: http://localhost:{API_PORT}/health'))
    print(Colors.dim(f'   Agents: http://localhost:{API_PORT}/agents'))
    print(Colors.dim(f'   Chat: http://localhost:{API_PORT}/chat'))
    print()

    # Start JAF server
    config_uvicorn = uvicorn.Config(app, host='127.0.0.1', port=API_PORT, log_level="info")
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(Colors.cyan('\n👋 Goodbye!'))
    except Exception as e:
        print(Colors.yellow(f'Error: {e}'))
        import traceback
        traceback.print_exc()