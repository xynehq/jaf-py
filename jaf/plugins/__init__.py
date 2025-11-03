"""
JAF Plugin System

This module provides a comprehensive plugin architecture for extending JAF
with custom tools, providers, and functionality. It supports dynamic loading,
dependency management, and lifecycle hooks.
"""

from .registry import PluginRegistry, get_plugin_registry
from .base import JAFPlugin, PluginMetadata, PluginStatus
from .loader import PluginLoader, load_plugins_from_directory
from .manager import PluginManager
from .decorators import plugin, tool_provider, model_provider, memory_provider

__all__ = [
    "JAFPlugin",
    "PluginMetadata",
    "PluginStatus",
    "PluginRegistry",
    "PluginLoader",
    "PluginManager",
    "get_plugin_registry",
    "load_plugins_from_directory",
    "plugin",
    "tool_provider",
    "model_provider",
    "memory_provider",
]
