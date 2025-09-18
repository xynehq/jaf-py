"""
File system agent for HITL demo.

This module provides the main agent with file system tools.
"""

import os
from pathlib import Path
from typing import Any

from jaf.core.types import Agent, ModelConfig, RunState
from .tools import (
    FileSystemContext, 
    list_files_tool, 
    read_file_tool, 
    delete_file_tool, 
    edit_file_tool
)


def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key and value and not os.getenv(key):
                        os.environ[key] = value


# Load environment variables
load_env_file()

# Environment configuration - prefer LiteLLM for flexibility
LITELLM_BASE_URL = os.getenv('LITELLM_BASE_URL') or os.getenv('LITELLM_URL', 'http://localhost:4000')
LITELLM_API_KEY = os.getenv('LITELLM_API_KEY', 'sk-demo') 
LITELLM_MODEL = os.getenv('LITELLM_MODEL', 'gpt-4o-mini')

# Fallback: Direct OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')


def get_instructions(state: RunState[FileSystemContext]) -> str:
    """Get agent instructions."""
    return """You are a helpful file system assistant working in a sandboxed directory.

Available operations:
- listFiles: List files and directories
- readFile: Read file contents  
- deleteFile: Delete a file
- editFile: Edit or create a file

if you are sent an image attachment, then process it and base you decision on it.
"""


# File system agent
file_system_agent: Agent[FileSystemContext, Any] = Agent(
    name="FileSystemAgent",
    instructions=get_instructions,
    tools=[list_files_tool, read_file_tool, delete_file_tool, edit_file_tool],
    model_config=ModelConfig(
        name=LITELLM_MODEL,
        temperature=0.1
    )
)