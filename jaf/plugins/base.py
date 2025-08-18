"""
Base plugin system components for JAF.

This module defines the core interfaces and data structures for the JAF plugin system,
including plugin metadata, lifecycle management, and extension points.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from enum import Enum
from datetime import datetime

from ..core.types import Tool, ModelProvider, Agent


class PluginStatus(str, Enum):
    """Plugin lifecycle status."""
    UNLOADED = 'unloaded'
    LOADING = 'loading'
    LOADED = 'loaded'
    ACTIVE = 'active'
    ERROR = 'error'
    DISABLED = 'disabled'


class PluginType(str, Enum):
    """Types of plugins supported by JAF."""
    TOOL_PROVIDER = 'tool_provider'
    MODEL_PROVIDER = 'model_provider'
    MEMORY_PROVIDER = 'memory_provider'
    AGENT_EXTENSION = 'agent_extension'
    MIDDLEWARE = 'middleware'
    INTEGRATION = 'integration'
    CUSTOM = 'custom'


@dataclass(frozen=True)
class PluginDependency:
    """Represents a plugin dependency."""
    name: str
    version_constraint: str = "*"  # Semantic version constraint
    optional: bool = False
    description: str = ""


@dataclass(frozen=True)
class PluginMetadata:
    """Comprehensive metadata for a JAF plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    entry_point: str
    dependencies: List[PluginDependency] = field(default_factory=list)
    jaf_version_constraint: str = "*"
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    configuration_schema: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'plugin_type': self.plugin_type.value,
            'entry_point': self.entry_point,
            'dependencies': [
                {
                    'name': dep.name,
                    'version_constraint': dep.version_constraint,
                    'optional': dep.optional,
                    'description': dep.description
                }
                for dep in self.dependencies
            ],
            'jaf_version_constraint': self.jaf_version_constraint,
            'license': self.license,
            'homepage': self.homepage,
            'repository': self.repository,
            'keywords': self.keywords,
            'configuration_schema': self.configuration_schema,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PluginContext:
    """Context provided to plugins during lifecycle operations."""
    plugin_metadata: PluginMetadata
    configuration: Dict[str, Any]
    jaf_version: str
    plugin_directory: Optional[str] = None
    logger: Optional[Any] = None
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.configuration.get(key, default)
    
    def has_config(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self.configuration


@runtime_checkable
class JAFPlugin(Protocol):
    """
    Base protocol for all JAF plugins.
    
    Plugins must implement this interface to be compatible with the JAF plugin system.
    The plugin system provides lifecycle management, dependency resolution, and
    configuration management.
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...
    
    async def initialize(self, context: PluginContext) -> None:
        """
        Initialize the plugin with the provided context.
        
        This method is called once when the plugin is loaded. Use this for
        one-time setup operations like establishing connections, loading
        configuration, or preparing resources.
        
        Args:
            context: Plugin context with configuration and metadata
        """
        ...
    
    async def activate(self) -> None:
        """
        Activate the plugin and make it available for use.
        
        This method is called after initialization to activate the plugin's
        functionality. Register tools, providers, or other extensions here.
        """
        ...
    
    async def deactivate(self) -> None:
        """
        Deactivate the plugin and clean up resources.
        
        This method is called when the plugin is being unloaded or the
        system is shutting down. Clean up connections, unregister handlers,
        and release resources here.
        """
        ...
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the current health status of the plugin.
        
        Returns:
            Dictionary containing health information including status,
            last check time, and any relevant metrics or error information.
        """
        ...


class BaseJAFPlugin:
    """
    Base implementation of JAFPlugin with common functionality.
    
    This class provides a foundation for creating JAF plugins with standard
    lifecycle management, health checking, and error handling.
    """
    
    def __init__(self, metadata: PluginMetadata):
        self._metadata = metadata
        self._status = PluginStatus.UNLOADED
        self._context: Optional[PluginContext] = None
        self._last_health_check = datetime.now()
        self._health_status = {'status': 'unknown', 'message': 'Not initialized'}
        self._error_count = 0
        self._last_error: Optional[str] = None
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata
    
    @property
    def status(self) -> PluginStatus:
        """Get current plugin status."""
        return self._status
    
    @property
    def context(self) -> Optional[PluginContext]:
        """Get plugin context (available after initialization)."""
        return self._context
    
    async def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with context."""
        try:
            self._status = PluginStatus.LOADING
            self._context = context
            
            # Validate configuration against schema if provided
            if self._metadata.configuration_schema:
                self._validate_configuration(context.configuration)
            
            # Call plugin-specific initialization
            await self._on_initialize(context)
            
            self._status = PluginStatus.LOADED
            self._health_status = {'status': 'healthy', 'message': 'Plugin loaded successfully'}
            
        except Exception as e:
            self._status = PluginStatus.ERROR
            self._error_count += 1
            self._last_error = str(e)
            self._health_status = {'status': 'error', 'message': f'Initialization failed: {e}'}
            raise
    
    async def activate(self) -> None:
        """Activate the plugin."""
        try:
            if self._status != PluginStatus.LOADED:
                raise RuntimeError(f"Cannot activate plugin in status: {self._status}")
            
            await self._on_activate()
            self._status = PluginStatus.ACTIVE
            self._health_status = {'status': 'healthy', 'message': 'Plugin active'}
            
        except Exception as e:
            self._status = PluginStatus.ERROR
            self._error_count += 1
            self._last_error = str(e)
            self._health_status = {'status': 'error', 'message': f'Activation failed: {e}'}
            raise
    
    async def deactivate(self) -> None:
        """Deactivate the plugin."""
        try:
            await self._on_deactivate()
            self._status = PluginStatus.LOADED
            self._health_status = {'status': 'inactive', 'message': 'Plugin deactivated'}
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            # Don't change status on deactivation error, but log it
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        self._last_health_check = datetime.now()
        
        # Get plugin-specific health info
        plugin_health = self._get_plugin_health()
        
        return {
            'plugin_name': self._metadata.name,
            'plugin_version': self._metadata.version,
            'status': self._status.value,
            'health': self._health_status,
            'last_check': self._last_health_check.isoformat(),
            'error_count': self._error_count,
            'last_error': self._last_error,
            'plugin_specific': plugin_health
        }
    
    # Abstract methods for plugin-specific implementation
    
    async def _on_initialize(self, context: PluginContext) -> None:
        """Plugin-specific initialization logic. Override in subclasses."""
        pass
    
    async def _on_activate(self) -> None:
        """Plugin-specific activation logic. Override in subclasses."""
        pass
    
    async def _on_deactivate(self) -> None:
        """Plugin-specific deactivation logic. Override in subclasses."""
        pass
    
    def _get_plugin_health(self) -> Dict[str, Any]:
        """Get plugin-specific health information. Override in subclasses."""
        return {'status': 'ok'}
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema. Override for custom validation."""
        # Basic validation - could be enhanced with jsonschema
        if not self._metadata.configuration_schema:
            return
        
        schema = self._metadata.configuration_schema
        required_fields = schema.get('required', [])
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required configuration field '{field}' is missing")


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for plugins that provide tools."""
    
    def get_tools(self) -> List[Tool[Any, Any]]:
        """Get list of tools provided by this plugin."""
        ...
    
    def get_tool_by_name(self, name: str) -> Optional[Tool[Any, Any]]:
        """Get a specific tool by name."""
        ...


@runtime_checkable
class ModelProviderPlugin(Protocol):
    """Protocol for plugins that provide model providers."""
    
    def get_model_provider(self) -> ModelProvider[Any]:
        """Get the model provider instance."""
        ...
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        ...


@runtime_checkable
class MemoryProviderPlugin(Protocol):
    """Protocol for plugins that provide memory providers."""
    
    def get_memory_provider(self) -> Any:  # MemoryProvider type
        """Get the memory provider instance."""
        ...
    
    def get_provider_capabilities(self) -> Dict[str, Any]:
        """Get capabilities and configuration options for the memory provider."""
        ...


@runtime_checkable
class AgentExtension(Protocol):
    """Protocol for plugins that extend agent functionality."""
    
    def extend_agent(self, agent: Agent[Any, Any]) -> Agent[Any, Any]:
        """Extend an agent with additional functionality."""
        ...
    
    def get_extension_info(self) -> Dict[str, Any]:
        """Get information about what this extension provides."""
        ...


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    
    def __init__(self, message: str, plugin_name: str = "", error_code: str = ""):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.error_code = error_code


class PluginLoadError(PluginError):
    """Exception raised when a plugin fails to load."""
    pass


class PluginDependencyError(PluginError):
    """Exception raised when plugin dependencies cannot be resolved."""
    pass


class PluginConfigurationError(PluginError):
    """Exception raised when plugin configuration is invalid."""
    pass


class PluginVersionError(PluginError):
    """Exception raised when plugin version constraints are not met."""
    pass
