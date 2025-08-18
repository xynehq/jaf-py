# JAF Documentation Hub

![JAF Banner](cover.png)

Welcome to the comprehensive documentation for JAF (Juspay Agent Framework) - Python Edition. This documentation will guide you through everything from basic concepts to advanced deployment scenarios.

## Quick Start

New to JAF? Start here:

1. **[Getting Started](getting-started.md)** - Installation, setup, and your first agent
2. **[Core Concepts](core-concepts.md)** - Understanding JAF's functional architecture
3. **[Examples](examples.md)** - Working examples and demos

## Documentation Structure

###  Foundation
- **[Getting Started](getting-started.md)** - Step-by-step setup guide
- **[Core Concepts](core-concepts.md)** - JAF's philosophy and architecture
- **[API Reference](api-reference.md)** - Complete Python API documentation

###  Building with JAF
- **[Tools](tools.md)** - Creating and using tools in Python
- **[Function Composition](function-composition.md)** - Advanced composition patterns
- **[Memory System](memory-system.md)** - Conversation persistence and memory providers
- **[Model Providers](model-providers.md)** - LiteLLM integration and configuration

###  Advanced Features
- **[Analytics System](analytics-system.md)** - Real-time monitoring and insights
- **[Workflow Orchestration](workflow-orchestration.md)** - Complex automation workflows
- **[Streaming Responses](streaming-responses.md)** - Real-time content delivery
- **[Performance Monitoring](performance-monitoring.md)** - System optimization and profiling

###  Production & Enterprise
- **[ADK Overview](adk-overview.md)** - Agent Development Kit for production
- **[A2A Deployment](a2a-deployment.md)** - Agent-to-Agent communication
- **[Server API](server-api.md)** - FastAPI server endpoints and usage
- **[Deployment](deployment.md)** - Docker, compose, and production setup
- **[Security Framework](security-framework.md)** - Enterprise security features

###  Learning & Examples
- **[Examples](examples.md)** - Detailed walkthrough of example applications
- **[Callback System](callback-system.md)** - Advanced agent instrumentation
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Use Case Navigation

### I want to...

**Build my first agent** → [Getting Started](getting-started.md)

**Understand JAF's architecture** → [Core Concepts](core-concepts.md)

**Create custom tools** → [Tools](tools.md)

**Add memory/persistence** → [Memory System](memory-system.md)

**Deploy to production** → [Server API](server-api.md) + [Deployment](deployment.md)

**Connect to different LLMs** → [Model Providers](model-providers.md)

**See working examples** → [Examples](examples.md)

**Solve a problem** → [Troubleshooting](troubleshooting.md)

## Framework Features

###  Complete Python Implementation
- Full feature parity with TypeScript version
- Type-safe with Pydantic models
- Immutable state management
- Comprehensive test coverage

###  Production Ready
- FastAPI server with auto-documentation
- Built-in security and validation
- Error handling and recovery
- Real-time tracing and observability

### 🔌 Extensive Integrations
- **Memory Providers**: In-memory, Redis, PostgreSQL
- **Model Providers**: LiteLLM (100+ models), OpenAI
- **MCP Support**: Model Context Protocol integration (Stdio & SSE transports)
- **Tool Ecosystem**: Extensible tool system

###  Developer Experience
- CLI tools for project management
- Hot reload development server
- Rich examples and tutorials
- Type hints throughout

## Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/xynehq/jaf-py/issues)
- **Examples**: Check the `examples/` directory for working code
- **API Reference**: Detailed function and class documentation
- **Troubleshooting**: Common solutions to frequent problems

## Contributing

JAF is open source and welcomes contributions! See our [Contributing Guide](../README.md#contributing) for details.

---

**Ready to build?** Start with [Getting Started](getting-started.md) or explore the [Examples](examples.md).
