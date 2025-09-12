"""
Shared utilities for JAF HITL demos.

This package provides reusable components for the HITL demonstrations.
"""

from .agent import file_system_agent
from .tools import (
    FileSystemContext,
    list_files_tool,
    read_file_tool,
    delete_file_tool,
    edit_file_tool,
    DEMO_DIR
)
from .memory import setup_memory_provider

__all__ = [
    'file_system_agent',
    'FileSystemContext',
    'list_files_tool',
    'read_file_tool', 
    'delete_file_tool',
    'edit_file_tool',
    'DEMO_DIR',
    'setup_memory_provider'
]