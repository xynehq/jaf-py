# Plugin System

JAF's plugin system provides a flexible architecture for extending framework capabilities through custom plugins, integrations, and extensions. This system enables seamless integration with external services, custom tools, and specialized functionality.

## Overview

The plugin system provides:

- **Modular Architecture**: Load and unload plugins dynamically
- **Standard Interfaces**: Consistent plugin development patterns
- **Lifecycle Management**: Plugin initialization, activation, and cleanup
- **Dependency Resolution**: Automatic plugin dependency management
- **Configuration Management**: Plugin-specific configuration and settings
- **Event System**: Plugin communication through events and hooks

## Core Components

### Plugin Interface

All plugins implement the base Plugin interface:

```python
from jaf.core.plugins import Plugin, PluginMetadata, PluginContext
from jaf.core.types import PluginConfig

class CustomPlugin(Plugin):
    """Example custom plugin implementation."""
    
    def __init__(self):
        super().__init__()
        self.metadata = PluginMetadata(
            name="custom_plugin",
            version="1.0.0",
            description="Example custom plugin",
            author="Plugin Developer",
            dependencies=["core", "tools"]
        )
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize plugin with context."""
        self.logger = context.logger
        self.config = context.config
        
        # Perform initialization
        self.logger.info(f"Initializing {self.metadata.name}")
        return True
    
    async def activate(self) -> bool:
        """Activate plugin functionality."""
        self.logger.info(f"Activating {self.metadata.name}")
        
        # Register tools, agents, or other functionality
        self._register_tools()
        self._register_event_handlers()
        
        return True
    
    async def deactivate(self) -> bool:
        """Deactivate plugin and cleanup resources."""
        self.logger.info(f"Deactivating {self.metadata.name}")
        
        # Cleanup resources
        self._cleanup_resources()
        
        return True
    
    def _register_tools(self):
        """Register plugin-specific tools."""
        pass
    
    def _register_event_handlers(self):
        """Register event handlers."""
        pass
    
    def _cleanup_resources(self):
        """Cleanup plugin resources."""
        pass
```

### Plugin Manager

The PluginManager handles plugin lifecycle and coordination:

```python
from jaf.core.plugins import PluginManager, PluginConfig

# Create plugin manager
plugin_manager = PluginManager()

# Load plugins from directory
await plugin_manager.load_plugins_from_directory("./plugins")

# Load specific plugin
await plugin_manager.load_plugin("custom_plugin", CustomPlugin())

# Get loaded plugins
loaded_plugins = plugin_manager.get_loaded_plugins()
print(f"Loaded plugins: {[p.metadata.name for p in loaded_plugins]}")

# Activate all plugins
await plugin_manager.activate_all()

# Get plugin by name
custom_plugin = plugin_manager.get_plugin("custom_plugin")
if custom_plugin:
    print(f"Plugin status: {custom_plugin.status}")
```

### Plugin Configuration

Plugins can define custom configuration schemas:

```python
from jaf.core.plugins import Plugin, PluginConfig
from pydantic import BaseModel

class DatabasePluginConfig(BaseModel):
    """Configuration schema for database plugin."""
    host: str = "localhost"
    port: int = 5432
    database: str = "jaf_db"
    username: str
    password: str
    pool_size: int = 10

class DatabasePlugin(Plugin):
    """Database integration plugin."""
    
    def __init__(self, config: DatabasePluginConfig):
        super().__init__()
        self.db_config = config
        self.connection_pool = None
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize database connection."""
        try:
            self.connection_pool = await self._create_connection_pool()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def _create_connection_pool(self):
        """Create database connection pool."""
        # Implementation would create actual connection pool
        return f"Pool({self.db_config.host}:{self.db_config.port})"
```

## Plugin Types

### Tool Plugins

Extend JAF with custom tools:

```python
from jaf.core.plugins import ToolPlugin
from jaf.core.tools import Tool

class WeatherToolPlugin(ToolPlugin):
    """Plugin that provides weather-related tools."""
    
    def __init__(self):
        super().__init__()
        self.metadata.name = "weather_tools"
        self.metadata.description = "Weather information tools"
    
    def get_tools(self) -> list[Tool]:
        """Return list of tools provided by this plugin."""
        return [
            self._create_weather_tool(),
            self._create_forecast_tool()
        ]
    
    def _create_weather_tool(self) -> Tool:
        """Create current weather tool."""
        @Tool(
            name="get_current_weather",
            description="Get current weather for a location"
        )
        def get_current_weather(location: str) -> dict:
            # Implementation would call weather API
            return {
                "location": location,
                "temperature": 72,
                "condition": "sunny",
                "humidity": 45
            }
        
        return get_current_weather
    
    def _create_forecast_tool(self) -> Tool:
        """Create weather forecast tool."""
        @Tool(
            name="get_weather_forecast",
            description="Get weather forecast for a location"
        )
        def get_weather_forecast(location: str, days: int = 5) -> dict:
            # Implementation would call forecast API
            return {
                "location": location,
                "forecast": [
                    {"day": i, "high": 75 + i, "low": 60 + i, "condition": "sunny"}
                    for i in range(days)
                ]
            }
        
        return get_weather_forecast
```

### Agent Plugins

Provide specialized agents:

```python
from jaf.core.plugins import AgentPlugin
from jaf import Agent

class CustomerServicePlugin(AgentPlugin):
    """Plugin providing customer service agents."""
    
    def get_agents(self) -> list[Agent]:
        """Return list of agents provided by this plugin."""
        return [
            self._create_support_agent(),
            self._create_billing_agent()
        ]
    
    def _create_support_agent(self) -> Agent:
        """Create general support agent."""
        def instructions(state):
            return """You are a helpful customer support agent.
            Assist customers with general inquiries and issues."""
        
        return Agent(
            name="GeneralSupportAgent",
            instructions=instructions,
            tools=self._get_support_tools()
        )
    
    def _create_billing_agent(self) -> Agent:
        """Create billing specialist agent."""
        def instructions(state):
            return """You are a billing specialist agent.
            Help customers with billing questions and payment issues."""
        
        return Agent(
            name="BillingAgent",
            instructions=instructions,
            tools=self._get_billing_tools()
        )
```

### Integration Plugins

Connect with external services:

```python
from jaf.core.plugins import IntegrationPlugin
import httpx

class SlackIntegrationPlugin(IntegrationPlugin):
    """Slack integration plugin."""
    
    def __init__(self, slack_token: str):
        super().__init__()
        self.slack_token = slack_token
        self.client = None
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize Slack client."""
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self.slack_token}"}
        )
        return True
    
    async def send_message(self, channel: str, message: str) -> bool:
        """Send message to Slack channel."""
        try:
            response = await self.client.post(
                "https://slack.com/api/chat.postMessage",
                json={
                    "channel": channel,
                    "text": message
                }
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}")
            return False
    
    def get_tools(self) -> list[Tool]:
        """Return Slack-related tools."""
        @Tool(
            name="send_slack_message",
            description="Send a message to a Slack channel"
        )
        async def send_slack_message(channel: str, message: str) -> dict:
            success = await self.send_message(channel, message)
            return {"success": success, "channel": channel}
        
        return [send_slack_message]
```

## Advanced Features

### Plugin Events

Plugins can communicate through events:

```python
from jaf.core.plugins import Plugin, PluginEvent, EventHandler

class EventDrivenPlugin(Plugin):
    """Plugin that uses event system."""
    
    async def activate(self) -> bool:
        """Activate plugin and register event handlers."""
        # Register event handlers
        self.register_event_handler("user_login", self._handle_user_login)
        self.register_event_handler("order_created", self._handle_order_created)
        
        return True
    
    @EventHandler("user_login")
    async def _handle_user_login(self, event: PluginEvent):
        """Handle user login event."""
        user_id = event.data.get("user_id")
        self.logger.info(f"User {user_id} logged in")
        
        # Emit follow-up event
        await self.emit_event("user_activity", {
            "user_id": user_id,
            "activity": "login",
            "timestamp": event.timestamp
        })
    
    @EventHandler("order_created")
    async def _handle_order_created(self, event: PluginEvent):
        """Handle order creation event."""
        order_id = event.data.get("order_id")
        self.logger.info(f"Order {order_id} created")
        
        # Process order
        await self._process_new_order(order_id)
```

### Plugin Dependencies

Manage plugin dependencies automatically:

```python
from jaf.core.plugins import Plugin, PluginDependency

class AdvancedPlugin(Plugin):
    """Plugin with dependencies."""
    
    def __init__(self):
        super().__init__()
        self.metadata.dependencies = [
            PluginDependency("database_plugin", ">=1.0.0"),
            PluginDependency("auth_plugin", ">=2.1.0"),
            PluginDependency("logging_plugin", ">=1.5.0", optional=True)
        ]
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize with dependency injection."""
        # Get required dependencies
        self.db_plugin = context.get_dependency("database_plugin")
        self.auth_plugin = context.get_dependency("auth_plugin")
        
        # Get optional dependency
        self.logging_plugin = context.get_dependency("logging_plugin", required=False)
        
        if not self.db_plugin or not self.auth_plugin:
            self.logger.error("Required dependencies not available")
            return False
        
        return True
```

### Plugin Hot Reloading

Reload plugins without restarting the application:

```python
from jaf.core.plugins import PluginManager

# Enable hot reloading
plugin_manager = PluginManager(enable_hot_reload=True)

# Reload specific plugin
await plugin_manager.reload_plugin("custom_plugin")

# Reload all plugins
await plugin_manager.reload_all_plugins()

# Watch for plugin file changes
plugin_manager.start_file_watcher("./plugins")
```

## Best Practices

### 1. Plugin Isolation

Ensure plugins don't interfere with each other:

```python
class IsolatedPlugin(Plugin):
    """Plugin with proper isolation."""
    
    def __init__(self):
        super().__init__()
        self._namespace = f"plugin_{self.metadata.name}"
        self._resources = []
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize with namespace isolation."""
        # Use namespaced configuration
        self.config = context.config.get_namespace(self._namespace)
        
        # Create isolated logger
        self.logger = context.logger.getChild(self._namespace)
        
        return True
    
    def _register_tool(self, tool):
        """Register tool with namespace."""
        namespaced_name = f"{self._namespace}_{tool.name}"
        tool.name = namespaced_name
        self._resources.append(tool)
```

### 2. Error Handling

Implement robust error handling:

```python
class RobustPlugin(Plugin):
    """Plugin with comprehensive error handling."""
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize with error handling."""
        try:
            await self._setup_resources()
            return True
        except Exception as e:
            self.logger.error(f"Plugin initialization failed: {e}")
            await self._cleanup_partial_setup()
            return False
    
    async def _setup_resources(self):
        """Setup plugin resources."""
        # Implementation with proper error handling
        pass
    
    async def _cleanup_partial_setup(self):
        """Cleanup resources if initialization fails."""
        # Implementation to cleanup partial setup
        pass
```

### 3. Configuration Validation

Validate plugin configuration:

```python
from pydantic import BaseModel, validator

class PluginConfig(BaseModel):
    """Plugin configuration with validation."""
    api_key: str
    timeout: int = 30
    max_retries: int = 3
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError('Timeout must be positive')
        return v
    
    @validator('max_retries')
    def validate_retries(cls, v):
        if v < 0:
            raise ValueError('Max retries cannot be negative')
        return v

class ValidatedPlugin(Plugin):
    """Plugin with configuration validation."""
    
    def __init__(self, config: PluginConfig):
        super().__init__()
        self.config = config  # Already validated by Pydantic
```

## Example: Complete Plugin Implementation

Here's a comprehensive example showing a complete plugin implementation:

```python
import asyncio
from typing import Dict, List, Optional
from jaf.core.plugins import Plugin, PluginManager, PluginMetadata, PluginContext
from jaf.core.tools import Tool
from jaf import Agent
from pydantic import BaseModel

class EmailPluginConfig(BaseModel):
    """Configuration for email plugin."""
    smtp_host: str
    smtp_port: int = 587
    username: str
    password: str
    use_tls: bool = True

class EmailPlugin(Plugin):
    """Comprehensive email plugin with tools and agents."""
    
    def __init__(self, config: EmailPluginConfig):
        super().__init__()
        self.config = config
        self.smtp_client = None
        
        self.metadata = PluginMetadata(
            name="email_plugin",
            version="1.0.0",
            description="Email functionality plugin",
            author="JAF Team",
            dependencies=["core"]
        )
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize email plugin."""
        try:
            self.logger = context.logger.getChild("email_plugin")
            self.logger.info("Initializing email plugin")
            
            # Initialize SMTP client
            await self._setup_smtp_client()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize email plugin: {e}")
            return False
    
    async def activate(self) -> bool:
        """Activate email plugin."""
        try:
            self.logger.info("Activating email plugin")
            
            # Register tools and agents
            self._register_tools()
            self._register_agents()
            self._register_event_handlers()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to activate email plugin: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """Deactivate email plugin."""
        try:
            self.logger.info("Deactivating email plugin")
            
            # Cleanup resources
            if self.smtp_client:
                await self.smtp_client.quit()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to deactivate email plugin: {e}")
            return False
    
    async def _setup_smtp_client(self):
        """Setup SMTP client."""
        # Implementation would create actual SMTP client
        self.smtp_client = f"SMTP({self.config.smtp_host}:{self.config.smtp_port})"
        self.logger.info("SMTP client initialized")
    
    def _register_tools(self):
        """Register email tools."""
        @Tool(
            name="send_email",
            description="Send an email message"
        )
        async def send_email(
            to: str,
            subject: str,
            body: str,
            cc: Optional[str] = None,
            bcc: Optional[str] = None
        ) -> Dict:
            """Send email using SMTP."""
            try:
                # Implementation would send actual email
                self.logger.info(f"Sending email to {to}: {subject}")
                
                return {
                    "success": True,
                    "message_id": f"msg_{hash(to + subject)}",
                    "recipients": [to] + (cc.split(',') if cc else [])
                }
            except Exception as e:
                self.logger.error(f"Failed to send email: {e}")
                return {"success": False, "error": str(e)}
        
        @Tool(
            name="send_template_email",
            description="Send email using a template"
        )
        async def send_template_email(
            to: str,
            template: str,
            variables: Dict
        ) -> Dict:
            """Send templated email."""
            try:
                # Implementation would render template and send
                rendered_subject = f"Template: {template}"
                rendered_body = f"Template {template} with variables: {variables}"
                
                return await send_email(to, rendered_subject, rendered_body)
            except Exception as e:
                self.logger.error(f"Failed to send template email: {e}")
                return {"success": False, "error": str(e)}
        
        # Register tools with plugin manager
        self.tools = [send_email, send_template_email]
    
    def _register_agents(self):
        """Register email-related agents."""
        def email_agent_instructions(state):
            return """You are an email assistant agent.
            Help users compose, send, and manage emails effectively.
            Use the available email tools to send messages."""
        
        email_agent = Agent(
            name="EmailAgent",
            instructions=email_agent_instructions,
            tools=self.tools
        )
        
        self.agents = [email_agent]
    
    def _register_event_handlers(self):
        """Register event handlers."""
        @self.event_handler("user_signup")
        async def handle_user_signup(self, event):
            """Send welcome email on user signup."""
            user_email = event.data.get("email")
            if user_email:
                await self.tools[1](  # send_template_email
                    to=user_email,
                    template="welcome",
                    variables={"name": event.data.get("name", "User")}
                )

async def main():
    """Demonstrate the email plugin system."""
    
    # Create plugin configuration
    email_config = EmailPluginConfig(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="your-email@gmail.com",
        password="your-password"
    )
    
    # Create plugin
    email_plugin = EmailPlugin(email_config)
    
    # Create plugin manager
    plugin_manager = PluginManager()
    
    # Load and activate plugin
    await plugin_manager.load_plugin("email_plugin", email_plugin)
    await plugin_manager.activate_plugin("email_plugin")
    
    # Use plugin tools
    email_tool = plugin_manager.get_tool("send_email")
    if email_tool:
        result = await email_tool(
            to="user@example.com",
            subject="Test Email",
            body="This is a test email from JAF plugin system"
        )
        print(f"Email result: {result}")
    
    # Use plugin agent
    email_agent = plugin_manager.get_agent("EmailAgent")
    if email_agent:
        print(f"Email agent available: {email_agent.name}")
    
    # Emit event to trigger email
    await plugin_manager.emit_event("user_signup", {
        "email": "newuser@example.com",
        "name": "New User"
    })
    
    # Cleanup
    await plugin_manager.deactivate_all()

if __name__ == "__main__":
    asyncio.run(main())
```

The plugin system provides a powerful foundation for extending JAF with custom functionality while maintaining clean separation of concerns and robust lifecycle management.

## Next Steps

- Learn about [Analytics System](analytics-system.md) for plugin monitoring
- Explore [Performance Monitoring](performance-monitoring.md) for plugin optimization
- Check [Streaming Responses](streaming-responses.md) for real-time plugin interactions
- Review [Workflow Orchestration](workflow-orchestration.md) for plugin coordination
