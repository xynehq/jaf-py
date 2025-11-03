#!/usr/bin/env python3
"""
Command-line interface for JAF (Juspay Agent Framework).

This module provides CLI commands for running and managing JAF servers.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from .core.types import Agent, RunConfig
from .providers.model import make_litellm_provider
from .server import run_server
from .server.types import ServerConfig


def create_default_config() -> ServerConfig:
    """Create a default server configuration."""

    # Create a simple echo agent for testing
    def echo_instructions(state) -> str:
        return """You are a helpful assistant that echoes back what users say.
        Simply repeat their message in a friendly way."""

    echo_agent = Agent(
        name="echo",
        instructions=echo_instructions,
        tools=None,
        output_codec=None,
        handoffs=None,
        model_config=None,
    )

    # Create model provider
    model_provider = make_litellm_provider(
        base_url="https://api.openai.com/v1", api_key="your-api-key-here"
    )

    # Create run config
    run_config = RunConfig(
        agent_registry={"echo": echo_agent}, model_provider=model_provider, max_turns=10
    )

    return ServerConfig(
        host="0.0.0.0", port=8000, agent_registry={"echo": echo_agent}, run_config=run_config
    )


async def run_server_command(args: argparse.Namespace) -> None:
    """Run the JAF server."""
    print("🚀 Starting JAF Server...")

    # Create server config
    config = create_default_config()

    # Override with command line arguments
    if args.host:
        config = config.__class__(**{**config.__dict__, "host": args.host})

    if args.port:
        config = config.__class__(**{**config.__dict__, "port": args.port})

    print(f"📍 Server will run on {config.host}:{config.port}")
    print(
        "💡 This is a default configuration. For production use, provide your own agents and model provider."
    )
    print("📚 See documentation for how to create custom agents and configurations.")

    try:
        await run_server(config)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="jaf", description="JAF (Juspay Agent Framework) - Command Line Interface"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run the JAF server")
    server_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to (default: 8000)"
    )
    server_parser.add_argument(
        "--config", type=str, help="Path to configuration file (not implemented yet)"
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    # Init command for creating project templates
    init_parser = subparsers.add_parser("init", help="Initialize a new JAF project")
    init_parser.add_argument("project_name", type=str, help="Name of the project to create")

    return parser


def show_version() -> None:
    """Show version information."""
    try:
        from . import __version__

        print(f"JAF (Juspay Agent Framework) version {__version__}")
    except ImportError:
        print("JAF (Juspay Agent Framework) version unknown")


def init_project(project_name: str) -> None:
    """Initialize a new JAF project."""
    project_path = Path(project_name)

    if project_path.exists():
        print(f"❌ Directory '{project_name}' already exists")
        sys.exit(1)

    print(f"📁 Creating JAF project: {project_name}")

    # Create project structure
    project_path.mkdir()
    (project_path / "agents").mkdir()
    (project_path / "tools").mkdir()

    # Create main.py
    main_py_content = '''#!/usr/bin/env python3
"""
JAF project main entry point.
"""

import asyncio
from jaf import run_server, Agent, RunConfig
from jaf.providers.model import make_litellm_provider
from jaf.server.types import ServerConfig

def my_agent_instructions(state):
    """Instructions for my custom agent."""
    return """You are a helpful assistant. 
    Respond to user queries in a friendly and informative way."""

def create_agents():
    """Create and return agent registry."""
    my_agent = Agent(
        name="my_agent",
        instructions=my_agent_instructions,
        tools=None,  # Add your tools here
        output_codec=None,
        handoffs=None,
        model_config=None
    )
    
    return {"my_agent": my_agent}

def create_server_config():
    """Create server configuration."""
    agents = create_agents()
    
    # Configure your model provider
    model_provider = make_litellm_provider(
        base_url="https://api.openai.com/v1",
        api_key="your-openai-api-key"  # Use environment variables in production
    )
    
    run_config = RunConfig(
        agent_registry=agents,
        model_provider=model_provider,
        max_turns=50
    )
    
    return ServerConfig(
        host="0.0.0.0",
        port=8000,
        agent_registry=agents,
        run_config=run_config
    )

async def main():
    """Main application entry point."""
    config = create_server_config()
    await run_server(config)

if __name__ == "__main__":
    asyncio.run(main())
'''

    (project_path / "main.py").write_text(main_py_content)

    # Create requirements.txt
    requirements_content = """jaf-py>=2.0.0
python-dotenv>=1.0.0
"""
    (project_path / "requirements.txt").write_text(requirements_content)

    # Create .env.example
    env_example_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Server Configuration
JAF_HOST=0.0.0.0
JAF_PORT=8000

# Model Configuration
JAF_MODEL=gpt-4o
JAF_TEMPERATURE=0.7
JAF_MAX_TOKENS=1000
"""
    (project_path / ".env.example").write_text(env_example_content)

    # Create README.md
    readme_content = f"""# {project_name}

A JAF (Juspay Agent Framework) project.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy environment file and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Run the server:
   ```bash
   python main.py
   ```

## Project Structure

- `main.py` - Main application entry point
- `agents/` - Directory for custom agent implementations
- `tools/` - Directory for custom tool implementations
- `.env` - Environment configuration (create from .env.example)

## Usage

Once the server is running, you can:

- View API documentation: http://localhost:8000/docs
- Check health: http://localhost:8000/health
- List agents: http://localhost:8000/agents
- Chat with agents: http://localhost:8000/chat

## Development

Edit `main.py` to:
- Add custom agents
- Configure model providers
- Add tools and capabilities
- Customize server settings
"""

    (project_path / "README.md").write_text(readme_content)

    print(f"✅ Project '{project_name}' created successfully!")
    print("\n📋 Next steps:")
    print(f"  cd {project_name}")
    print("  pip install -r requirements.txt")
    print("  cp .env.example .env")
    print("  # Edit .env with your configuration")
    print("  python main.py")


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "server":
        await run_server_command(args)
    elif args.command == "version":
        show_version()
    elif args.command == "init":
        init_project(args.project_name)
    else:
        parser.print_help()


def cli_main() -> None:
    """Synchronous entry point for CLI."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
